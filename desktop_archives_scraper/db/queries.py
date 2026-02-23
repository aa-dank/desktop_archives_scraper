from __future__ import annotations

from datetime import datetime
from typing import Sequence

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from desktop_archives_scraper.db.models import FileContent, FileContentFailure


def persist_processing_batch(
	session: Session,
	*,
	content_rows: Sequence[dict],
	failure_rows: Sequence[dict],
) -> tuple[int, int, int]:
	"""
	Persist a worker batch using idempotent upserts.

	Parameters
	----------
	session:
		Active SQLAlchemy session.
	content_rows:
		Rows for `file_contents` upsert.
	failure_rows:
		Rows for `file_content_failures` upsert.

	Returns
	-------
	tuple[int, int, int]
		(content_upserts, failure_upserts, failures_cleared)
	"""
	content_upserts = 0
	failure_upserts = 0
	failures_cleared = 0

	if content_rows:
		stmt = insert(FileContent).values(list(content_rows))
		stmt = stmt.on_conflict_do_update(
			index_elements=[FileContent.file_hash],
			set_={
				"source_text": stmt.excluded.source_text,
				"minilm_model": stmt.excluded.minilm_model,
				"minilm_emb": stmt.excluded.minilm_emb,
				"mpnet_model": stmt.excluded.mpnet_model,
				"mpnet_emb": stmt.excluded.mpnet_emb,
				"updated_at": stmt.excluded.updated_at,
				"text_length": stmt.excluded.text_length,
			},
		)
		session.execute(stmt)
		content_upserts = len(content_rows)

		successful_hashes = [row["file_hash"] for row in content_rows]
		if successful_hashes:
			failures_cleared = (
				session.query(FileContentFailure)
				.filter(FileContentFailure.file_hash.in_(successful_hashes))
				.delete(synchronize_session=False)
			)

	if failure_rows:
		stmt = insert(FileContentFailure).values(list(failure_rows))
		stmt = stmt.on_conflict_do_update(
			index_elements=[FileContentFailure.file_hash],
			set_={
				"stage": stmt.excluded.stage,
				"error": stmt.excluded.error,
				"attempts": FileContentFailure.attempts + 1,
				"last_failed_at": stmt.excluded.last_failed_at,
			},
		)
		session.execute(stmt)
		failure_upserts = len(failure_rows)

	session.commit()
	return content_upserts, failure_upserts, failures_cleared


def failure_row(
	*,
	file_hash: str,
	stage: str,
	error: str,
	failed_at: datetime,
) -> dict:
	return {
		"file_hash": file_hash,
		"stage": stage,
		"error": error,
		"attempts": 1,
		"last_failed_at": failed_at,
	}

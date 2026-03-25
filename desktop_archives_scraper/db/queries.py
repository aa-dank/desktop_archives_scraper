from __future__ import annotations

from datetime import datetime
from typing import Sequence

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from desktop_archives_scraper.db.models import FileContent, FileContentFailure, FileDateMention


def persist_processing_batch(
	session: Session,
	*,
	content_rows: Sequence[dict],
	failure_rows: Sequence[dict],
	date_mention_rows: Sequence[dict],
	replace_date_mentions_for_hashes: Sequence[str] = (),
) -> tuple[int, int, int, int]:
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
	date_mention_rows:
		Rows for `file_date_mentions` replace-all upsert.
	replace_date_mentions_for_hashes:
		Successful file hashes whose existing date mentions should be deleted
		before inserting newly extracted rows.

	Returns
	-------
	tuple[int, int, int, int]
		(content_upserts, failure_upserts, failures_cleared, date_mention_upserts)
	"""
	content_upserts = 0
	failure_upserts = 0
	failures_cleared = 0
	date_mention_upserts = 0

	if content_rows:
		# Keep the last row for each file_hash so one INSERT statement never
		# proposes duplicate conflict keys.
		deduped_content_rows = list({row["file_hash"]: row for row in content_rows}.values())
		stmt = insert(FileContent).values(deduped_content_rows)
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
		content_upserts = len(deduped_content_rows)

		successful_hashes = [row["file_hash"] for row in deduped_content_rows]
		if successful_hashes:
			failures_cleared = (
				session.query(FileContentFailure)
				.filter(FileContentFailure.file_hash.in_(successful_hashes))
				.delete(synchronize_session=False)
			)

	replace_hashes = list(dict.fromkeys(replace_date_mentions_for_hashes))
	if replace_hashes:
		(
			session.query(FileDateMention)
			.filter(FileDateMention.file_hash.in_(replace_hashes))
			.delete(synchronize_session=False)
		)

	if date_mention_rows:
		stmt = insert(FileDateMention).values(list(date_mention_rows))
		stmt = stmt.on_conflict_do_update(
			index_elements=[
				FileDateMention.file_hash,
				FileDateMention.mention_date,
				FileDateMention.granularity,
			],
			set_={
				"mentions_count": stmt.excluded.mentions_count,
				"extractor": stmt.excluded.extractor,
				"extracted_at": stmt.excluded.extracted_at,
			},
		)
		session.execute(stmt)
		date_mention_upserts = len(date_mention_rows)

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
	return content_upserts, failure_upserts, failures_cleared, date_mention_upserts


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

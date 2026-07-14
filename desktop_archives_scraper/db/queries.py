from __future__ import annotations

from datetime import datetime
from typing import Sequence

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from desktop_archives_scraper.db.models import (
	FileContent,
	FileContentFailure,
	FileContentFtsChunk,
	FileDateMention,
)


def persist_processing_batch(
	session: Session,
	*,
	content_rows: Sequence[dict],
	fts_chunk_rows: Sequence[dict],
	failure_rows: Sequence[dict],
	date_mention_rows: Sequence[dict],
	replace_date_mentions_for_hashes: Sequence[str] = (),
) -> tuple[int, int, int, int, int, int]:
	"""
	Persist a worker batch using idempotent upserts.

	Parameters
	----------
	session:
		Active SQLAlchemy session.
	content_rows:
		Rows for `file_contents` upsert.
	fts_chunk_rows:
		Rows for `file_content_fts_chunks` rebuild.
	failure_rows:
		Rows for `file_content_failures` upsert.
	date_mention_rows:
		Rows for `file_date_mentions` replace-all upsert.
	replace_date_mentions_for_hashes:
		Successful file hashes whose existing date mentions should be deleted
		before inserting newly extracted rows.

	Returns
	-------
	tuple[int, int, int, int, int, int]
		(
			content_upserts,
			failure_upserts,
			failures_cleared,
			date_mention_upserts,
			fts_chunk_upserts,
			fts_chunk_files_rebuilt,
		)
	"""
	content_upserts = 0
	failure_upserts = 0
	failures_cleared = 0
	date_mention_upserts = 0
	fts_chunk_upserts = 0
	fts_chunk_files_rebuilt = 0

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
				"updated_at": stmt.excluded.updated_at,
				"text_length": stmt.excluded.text_length,
				"source_metadata": stmt.excluded.source_metadata,
			},
		)
		session.execute(stmt)
		content_upserts = len(deduped_content_rows)

		successful_hashes = [row["file_hash"] for row in deduped_content_rows]
		if successful_hashes:
			successful_hash_set = set(successful_hashes)
			failures_cleared = (
				session.query(FileContentFailure)
				.filter(FileContentFailure.file_hash.in_(successful_hashes))
				.delete(synchronize_session=False)
			)
			fts_chunk_files_rebuilt = len(successful_hashes)
			(
				session.query(FileContentFtsChunk)
				.filter(FileContentFtsChunk.file_hash.in_(successful_hashes))
				.delete(synchronize_session=False)
			)

			chunk_sets_by_hash: dict[str, dict] = {}
			for row in fts_chunk_rows:
				file_hash = row["file_hash"]
				chunked_at = row["chunked_at"]
				chunk_index = row["chunk_index"]
				existing_chunk_set = chunk_sets_by_hash.get(file_hash)
				if existing_chunk_set is None or chunked_at > existing_chunk_set["chunked_at"]:
					chunk_sets_by_hash[file_hash] = {
						"chunked_at": chunked_at,
						"rows_by_index": {chunk_index: row},
					}
					continue
				if chunked_at == existing_chunk_set["chunked_at"]:
					existing_chunk_set["rows_by_index"][chunk_index] = row

			deduped_chunk_rows = [
				row
				for file_hash, chunk_set in chunk_sets_by_hash.items()
				if file_hash in successful_hash_set
				for _, row in sorted(chunk_set["rows_by_index"].items())
			]
			if deduped_chunk_rows:
				session.execute(insert(FileContentFtsChunk).values(deduped_chunk_rows))
				fts_chunk_upserts = len(deduped_chunk_rows)

	replace_hashes = list(dict.fromkeys(replace_date_mentions_for_hashes))
	if replace_hashes:
		(
			session.query(FileDateMention)
			.filter(FileDateMention.file_hash.in_(replace_hashes))
			.delete(synchronize_session=False)
		)

	if date_mention_rows:
		# Multiple file records can refer to the same content hash. Their
		# extraction results may therefore contribute the same date mention to a
		# single flush. PostgreSQL cannot apply an ON CONFLICT update twice to
		# the same target row within one INSERT, so retain the latest result for
		# each primary-key tuple before issuing the upsert.
		deduped_date_mention_rows = list(
			{
				(
					row["file_hash"],
					row["mention_date"],
					row["granularity"],
				): row
				for row in date_mention_rows
			}.values()
		)
		stmt = insert(FileDateMention).values(deduped_date_mention_rows)
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
		date_mention_upserts = len(deduped_date_mention_rows)

	if failure_rows:
		stmt = insert(FileContentFailure).values(list(failure_rows))
		stmt = stmt.on_conflict_do_update(
			index_elements=[FileContentFailure.file_hash],
			set_={
				"stage": stmt.excluded.stage,
				"error": stmt.excluded.error,
				"attempts": FileContentFailure.attempts + 1,
				"last_failed_at": stmt.excluded.last_failed_at,
				"source_metadata": stmt.excluded.source_metadata,
			},
		)
		session.execute(stmt)
		failure_upserts = len(failure_rows)

	session.commit()
	return (
		content_upserts,
		failure_upserts,
		failures_cleared,
		date_mention_upserts,
		fts_chunk_upserts,
		fts_chunk_files_rebuilt,
	)


def failure_row(
	*,
	file_hash: str,
	stage: str,
	error: str,
	failed_at: datetime,
	source_metadata: dict,
) -> dict:
	return {
		"file_hash": file_hash,
		"stage": stage,
		"error": error,
		"attempts": 1,
		"last_failed_at": failed_at,
		"source_metadata": source_metadata,
	}

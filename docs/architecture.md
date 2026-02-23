# Architecture

## Purpose

`desktop_archives_scraper` is a Windows desktop execution variant of `archives_scraper` for enterprise backlog catch-up. It preserves the same production DB contract and failure semantics while reducing write churn through batch persistence.

## System Flow

1. CLI initializes logging, extractors, embedder, and DB session factory.
2. Worker polls `files`/`file_locations` for records missing `file_contents`.
3. Extraction pipeline copies each file to a temp path and runs extension-specific extraction.
4. Optional embedding generation (MiniLM).
5. Results are buffered and flushed in batched DB upserts.
6. Failures are upserted to `file_content_failures`; successes clear failure rows for those hashes.

## Reuse Map

| Module | Source | Notes |
|---|---|---|
| `desktop_archives_scraper/cli.py` | `archives_scraper/core/cli.py` + new | Preserves CLI semantics; adds batch tuning options for desktop runs. |
| `desktop_archives_scraper/worker.py` | `archives_scraper/core/worker.py` + new | Same processing/failure model, with batched persistence orchestration. |
| `desktop_archives_scraper/db/models.py` | `archives_scraper/db/models.py` | Canonical production table mapping. |
| `desktop_archives_scraper/db/db.py` | `archives_scraper/db/db.py` | Engine/session baseline. |
| `desktop_archives_scraper/db/queries.py` | new | Batch upsert helpers for `file_contents` and `file_content_failures`. |
| `desktop_archives_scraper/text_extraction/*` | `archives_scraper/text_extraction/*` | Robust extractors incl. large-file and subprocess guards. |
| `desktop_archives_scraper/embedding/*` | `archives_scraper/embedding/*` | MiniLM embedder path parity. |
| `desktop_archives_scraper/config.py` | new | Centralized runtime knob defaults from env. |

## Non-goals

- Date tagging or filing-tag pipelines from `file_code_tagger`.
- Schema migrations or DB shape changes.
- Tika-based fallback extraction.

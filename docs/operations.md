# Operations

## Multi-Instance Runbook (Desktop + Linux)

Target coexistence pattern:

- 1 Linux `archives_scraper` worker set
- 3–4 Windows desktop `desktop_archives_scraper` instances

## Recommended Startup Sequence

1. Validate DB connectivity from each desktop with `--limit 1 --dry-run`.
2. Start one desktop instance and observe logs for 5–10 minutes.
3. Add additional desktops gradually (one at a time).

## Suggested Baseline Knobs

- `POLL_BATCH_SIZE=10`
- `WRITE_BATCH_SIZE=25`
- `COMMIT_INTERVAL_SECONDS=5`
- `POLL_SECONDS=5`

For stronger desktops and stable DB latency, increase `WRITE_BATCH_SIZE` first before increasing `POLL_BATCH_SIZE`.

## Monitoring Signals

- `ok`, `error`, and `no_extractor` counts per batch
- Frequency of failure upserts in `file_content_failures`
- Volume of `file_content_fts_chunks` rebuilds during normal scraping
- Volume of `file_date_mentions` refreshes during normal scraping
- Unexpected spikes in duplicate/retry behavior

## Subsystem Resilience

- The worker has built-in connection retry logic for tracking broken DB connections (`sqlalchemy.exc.OperationalError`). If DB connection is briefly lost, the session connection drops, wait exponentially to reconnect up to 3 times before failing the run.
- Tuning DB engine pool behavior via `DB_CONNECT_TIMEOUT_SECONDS` and `DB_POOL_RECYCLE_SECONDS` provides further flexibility when deployed in restrictive network configurations.

## Failure Semantics

- Extraction/embedding failures are persisted with stage and attempt count.
- Successful content upsert clears existing failure rows for that file hash.
- Successful content upsert replaces existing `file_content_fts_chunks` rows for that file hash with a fresh chunk set derived from the latest extracted text.
- When date extraction is enabled, successful content upsert replaces any existing `file_date_mentions` rows for that file hash with the newly extracted set.
- Known concurrency duplicates are absorbed by upsert behavior rather than treated as hard failures.

## Troubleshooting

- Path issues: verify `FILE_SERVER_MOUNT` maps to the same server root semantics as `file_locations.file_server_directories`. Windows extended-length paths (`\\?\`) are natively handled by the worker to bypass the 260-character `MAX_PATH` limitation.
- Targeting specific files: Use `--hashes` CLI flag or `TARGET_HASHES` env var to limit processing to specific file hashes. Helpful for re-processing failed files. If targeting already-processed files, delete them from `file_contents` first.
- Tesseract issues: verify `C:\Program Files\Tesseract-OCR\tesseract.exe` exists, or set `TESSERACT_CMD` to the real executable path.
- OCR crashes: tune extractor `*_SUBPROCESS_*` and memory knobs.
- Slow writes: increase `WRITE_BATCH_SIZE` and/or `COMMIT_INTERVAL_SECONDS` carefully.

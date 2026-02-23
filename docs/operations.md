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
- Unexpected spikes in duplicate/retry behavior

## Failure Semantics

- Extraction/embedding failures are persisted with stage and attempt count.
- Successful content upsert clears existing failure rows for that file hash.
- Known concurrency duplicates are absorbed by upsert behavior rather than treated as hard failures.

## Troubleshooting

- Path issues: verify `FILE_SERVER_MOUNT` maps to the same server root semantics as `file_locations.file_server_directories`.
- OCR crashes: tune extractor `*_SUBPROCESS_*` and memory knobs.
- Slow writes: increase `WRITE_BATCH_SIZE` and/or `COMMIT_INTERVAL_SECONDS` carefully.

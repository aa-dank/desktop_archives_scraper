# Configuration

## Dotenv Support

- CLI loads `.env` automatically at startup via `python-dotenv`.
- Existing shell environment variables are not overwritten by `.env` values.
- Recommended setup: copy `.env.example` to `.env` and edit values locally.

## Required Environment Variables

- `DB_HOST`
- `DB_PORT`
- `DB_NAME`
- `DB_USERNAME`
- `DB_PASSWORD`
- `FILE_SERVER_MOUNT` (desktop-visible mount root that corresponds to `file_locations.file_server_directories`)

## Optional Runtime Variables

- `LIMIT`
- `POLL_SECONDS` (default `5.0`)
- `POLL_BATCH_SIZE` (default `10`)
- `WRITE_BATCH_SIZE` (default `25`)
- `COMMIT_INTERVAL_SECONDS` (default `5.0`)
- `EXTENSIONS` (comma-separated)
- `MAX_CHARS`
- `ENABLE_EMBEDDING` (`true`/`false`)
- `EMBEDDER` (`minilm`)
- `FAILURE_RETRY_TRESHOLD` (int; include failed files only when attempts are below this value)
- `RANDOMIZE` (`true`/`false`)
- `LOG_LEVEL`, `LOG_FILE`, `JSON_LOGS`

## Extraction-Specific Optional Variables

Inherited from extractor modules (use only if needed for tuning):

- PDF OCR controls (`OCR_*`)
- Image OCR controls (`IMAGE_OCR_*`)
- Office subprocess controls (`OFFICE_*`)

## CLI Examples

```powershell
uv run python -m desktop_archives_scraper.cli --help
uv run python -m desktop_archives_scraper.cli --limit 100 --poll-batch-size 20 --write-batch-size 50
uv run python -m desktop_archives_scraper.cli --extensions pdf,docx,pptx --failure_retry_treshold 2
```

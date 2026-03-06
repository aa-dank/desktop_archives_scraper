# desktop_archives_scraper

Temporary high-throughput Windows desktop ingestion worker that writes to the same PostgreSQL schema used by `archives_scraper`.

This repo exists to accelerate historical backlog processing while Linux production ingestion continues in parallel.

## Scope

- Parity target: same logical DB behavior as `archives_scraper` for `file_contents` and `file_content_failures`.
- No schema changes.
- No Tika.
- Date extraction/tagging is out of scope for v1.

## Architecture Sources

- Primary source of truth: `archives_scraper`
- Selective utility inspiration: `file_code_tagger`
- New code in this repo: package wiring + batched persistence tuning for desktop concurrency.

See `docs/architecture.md` for the detailed reuse map.

## Prerequisites

The following non-Python tools must be installed on each machine running the scraper:

| Tool | Required for | Notes |
|---|---|---|
| **Microsoft Word** | `.doc` extraction | Used via Windows COM (`Word.Application`). No LibreOffice needed. |
| **Microsoft PowerPoint** | `.ppt`, `.pps` extraction | Used via Windows COM (`PowerPoint.Application`). No LibreOffice needed. |
| **Tesseract OCR** | Image OCR (`.png`, `.jpg`, etc.) and PDF OCR fallback | Install from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki). Ensure `tesseract.exe` is on `PATH`, or set `TESSERACT_CMD` env var to the full path (e.g. `C:\Program Files\Tesseract-OCR\tesseract.exe`). |
| **Ghostscript** | PDF OCR (required by `ocrmypdf`) | Install from [ghostscript.com](https://www.ghostscript.com/). Ensure `gswin64c.exe` is on `PATH`. |

## Quick Start

1. Install dependencies:

	```powershell
	uv sync
	```

2. Create a local env file:

	```powershell
	Copy-Item .env.example .env
	```

3. Set required values in `.env` (or export in shell). PowerShell example:

	```powershell
	$env:DB_HOST = "..."
	$env:DB_PORT = "5432"
	$env:DB_NAME = "..."
	$env:DB_USERNAME = "..."
	$env:DB_PASSWORD = "..."
	$env:FILE_SERVER_MOUNT = "N:\\PPDO\\Records"
	```

4. Verify CLI:

	```powershell
	uv run python -m desktop_archives_scraper.cli --help
	```

5. Run a smoke batch:

	```powershell
	uv run python -m desktop_archives_scraper.cli --limit 25 --log-level INFO
	```

## Runtime Knobs

- `POLL_BATCH_SIZE`: files fetched per DB poll (default `10`)
- `WRITE_BATCH_SIZE`: processed file payloads buffered before DB flush (default `25`)
- `COMMIT_INTERVAL_SECONDS`: max seconds before forced flush of pending writes (default `5.0`)
- `POLL_SECONDS`: sleep between loops (default `5.0`)
- `MAX_RUNTIME_SECONDS`: wall-clock seconds after which the worker exits cleanly. Any in-progress file (including OCR) finishes first; pending DB writes are flushed before exit. Poll/idle sleeps are automatically capped to the remaining budget so overshoot is bounded to at most one file's processing time.

These can be set with env vars or CLI options (`--max-runtime-seconds`, etc.).

## Operational Notes

- Expected concurrent operation: Linux scraper + 3–4 desktop instances.
- Worker writes use idempotent upserts to reduce duplicate-write noise.
- Successful content upsert clears any existing failure row for that file hash.

See `docs/operations.md` for multi-instance runbook guidance.

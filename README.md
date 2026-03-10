# desktop_archives_scraper

Temporary high-throughput Windows desktop ingestion worker that writes to the same PostgreSQL schema used by `archives_scraper`.

This repo exists to accelerate historical backlog processing while Linux production ingestion continues in parallel.

## Architecture Sources

- Primary source of truth: `archives_scraper`
- Selective utility inspiration: `file_code_tagger`
- New code in this repo: package wiring + batched persistence tuning for desktop concurrency.

See `docs/architecture.md` for the detailed reuse map.

## Prerequisites

The following must be installed on each Windows machine running the scraper:

| Tool | Required for | Notes |
|---|---|---|
| **Python 3.13+** | Runtime | See install steps below. |
| **uv** | Dependency management | See install steps below. |
| **Microsoft Word** | `.doc` extraction | Used via Windows COM (`Word.Application`). No LibreOffice needed. |
| **Microsoft PowerPoint** | `.ppt`, `.pps` extraction | Used via Windows COM (`PowerPoint.Application`). No LibreOffice needed. |
| **Tesseract OCR** | Image OCR (`.png`, `.jpg`, etc.) and PDF OCR fallback | See install steps below. |
| **Ghostscript** | PDF OCR (required by `ocrmypdf`) | See install steps below. |

### Installing Python 3.13+

Download the latest Python 3.13 installer from [python.org](https://www.python.org/downloads/windows/) and run it.
Check **"Add python.exe to PATH"** during setup.

Verify:

```powershell
python --version
```

### Installing uv

Run the official installer in PowerShell (does not require admin):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Alternatively, install via winget:

```powershell
winget install --id=astral-sh.uv -e
```

Verify:

```powershell
uv --version
```

### Installing Tesseract OCR

1. Download the latest Windows installer from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki) (choose the `tesseract-ocr-w64-setup-*.exe` file).
2. Run the installer. The default install path is `C:\Program Files\Tesseract-OCR\`.
3. Add Tesseract to your system `PATH`:
   - Open **System Properties → Environment Variables**.
   - Under **System variables**, edit `Path` and add `C:\Program Files\Tesseract-OCR`.
4. Verify:

```powershell
tesseract --version
```

If you prefer not to modify `PATH`, set the `TESSERACT_CMD` environment variable instead:

```powershell
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### Installing Ghostscript

1. Download the latest Windows 64-bit installer from [ghostscript.com](https://www.ghostscript.com/download/gsdnld.html) (choose `gs*w64.exe`).
2. Run the installer (requires admin). The default install path is `C:\Program Files\gs\gs*\bin\`.
3. Add Ghostscript to your system `PATH`:
   - Open **System Properties → Environment Variables**.
   - Under **System variables**, edit `Path` and add the `bin` directory, e.g. `C:\Program Files\gs\gs10.05.1\bin`.
4. Verify:

```powershell
gswin64c --version
```

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
- Date mention extraction is enabled by default and refreshes `file_date_mentions` for successfully processed files.
- Use `--no-date-extract` to skip date mention extraction for a run without deleting existing date rows.
- The worker exits with code `3` on transient DB connection failures (e.g. the PostgreSQL server restarted). A simple restart wrapper script handles automatic recovery — see `run_with_retry.ps1` in the repo root.

See `docs/operations.md` for multi-instance runbook guidance.

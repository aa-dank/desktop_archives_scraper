# Repository Guidelines

## Project Structure & Module Organization

`desktop_archives_scraper/` contains the Python package and CLI entry point.
Core worker orchestration lives in `worker.py` and `cli.py`, configuration in `config.py`, database access in `db/`, embeddings in `embedding/`, and document/OCR parsing in `text_extraction/`.
Operational references live in `docs/`, including architecture, configuration, schema, testing, and runbook notes.
Windows deployment helpers live in `deployment_scripts/`.
Keep local secrets in `.env`; use `.env.example` as the committed template.

## Build, Test, and Development Commands

Run `uv sync` to install Python 3.13 dependencies from `pyproject.toml` and `uv.lock`.
Run `uv run python -m desktop_archives_scraper.cli --help` to verify the CLI loads.
Run `uv run python -m desktop_archives_scraper.cli --limit 5 --dry-run --log-level INFO` for a local smoke check without database writes.
Run `uv run python -m desktop_archives_scraper.cli --limit 25 --log-level INFO` for a small DB-backed smoke batch after environment variables are configured.
Use `deployment_scripts\Run-App.ps1` and related scripts only for Windows runtime/deployment workflows.

## Coding Style & Naming Conventions

Use Python 3.13 syntax and follow existing module style.
Use 4-space indentation in Python files, snake_case for modules, functions, variables, and CLI options, and PascalCase for classes.
Prefer typed, explicit configuration and small helper functions over large procedural blocks.
Keep extraction-specific logic inside `text_extraction/`, persistence logic inside `db/`, and CLI parsing inside `cli.py`.

## Testing Guidelines

There is no committed unit test suite yet.
Follow `docs/testing.md` for validation before merging changes.
For behavior changes, run the CLI help check and the dry-run smoke command at minimum.
For database, batching, extraction, or embedding changes, also run a small DB smoke batch and compare expected rows in `file_contents`, `file_date_mentions`, and `file_content_failures`.
Name future tests after the behavior under test, for example `test_pdf_extraction_records_failure.py`.

## Commit & Pull Request Guidelines

Recent commits use short, imperative summaries, often mentioning the changed behavior or documentation.
Keep commit messages concise, for example `document tesseract path resolution` or `add targeted hash scraping`.
Pull requests should describe the runtime impact, list validation commands run, and call out any configuration, schema, or deployment-script changes.
Include logs or screenshots only when they clarify a Windows runtime, OCR, or deployment issue.

## Security & Configuration Tips

Never commit `.env`, database credentials, file server paths, or production logs.
Document new environment variables in `.env.example` and `docs/configuration.md`.
Keep network and live database checks minimal, read-only where possible, and scoped to the validation needed for the change.

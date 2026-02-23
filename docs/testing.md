# Testing

## Local Validation Checklist

1. Dependency sync:

	```powershell
	uv sync
	```

2. CLI help:

	```powershell
	uv run python -m desktop_archives_scraper.cli --help
	```

3. Dry-run smoke:

	```powershell
	uv run python -m desktop_archives_scraper.cli --limit 5 --dry-run --log-level INFO
	```

4. DB smoke run:

	```powershell
	uv run python -m desktop_archives_scraper.cli --limit 25 --log-level INFO
	```

## Concurrent Validation (Small Scale)

Run two desktop instances simultaneously with different `--log-file` values and verify:

- Stable processing without systemic duplicate-write errors.
- Failure rows only for genuine extraction/embed errors.
- Materially fewer commits versus per-file commit behavior.

## Parity Validation

For a sample set of file hashes, compare outcomes against `archives_scraper` expectations:

- Presence/shape of `file_contents`
- Embedding fields populated when enabled
- `file_content_failures` lifecycle (insert, increment attempts, clear on success)

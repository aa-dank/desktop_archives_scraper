# Schema

This project uses the existing production PostgreSQL schema from `archives_scraper`.

## Key Tables Used

- `files`
- `file_locations`
- `file_contents`
- `file_content_failures`

## Contract Notes

- `file_contents.file_hash` is the primary key and is upserted for idempotency.
- `file_content_failures.file_hash` is a single active failure row per file hash.
- On successful content upsert, corresponding failure row is deleted.

## Non-Negotiable

- No new tables
- No column changes
- No index changes
- No migration files in v1

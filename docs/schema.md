# Schema

This project uses the existing production PostgreSQL schema from `archives_scraper`.

## Key Tables Used

- `files`
- `file_locations`
- `file_contents`
- `file_content_failures`
- `file_date_mentions`

## Contract Notes

- `file_contents.file_hash` is the primary key and is upserted for idempotency.
- `file_content_failures.file_hash` is a single active failure row per file hash.
- On successful content upsert, corresponding failure row is deleted.
- `file_date_mentions` is keyed by `(file_hash, mention_date, granularity)`.
- When date extraction is enabled, successful file processing replaces all existing `file_date_mentions` rows for that file hash with the newly extracted set.

## Non-Negotiable

- No repo-managed migrations
- No migration files in v1

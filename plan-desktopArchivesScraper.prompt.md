## Desktop Archives Scraper: Agent-Build Plan

## 1) Enterprise Vision and Project Role

This repository (`desktop_archives_scraper`) is a temporary high-throughput acceleration layer for enterprise ingestion. It exists to rapidly scrape the historical backlog from the mounted file server into the same PostgreSQL tables used by the Linux scraper, while remaining compatible with ongoing production ingestion.

### Domain Context (Important for Architecture Decisions)

The file corpus represents UC Santa Cruz capital project records across multiple eras and media qualities:

- Archival material (including scans from the 1960s era and later) with variable OCR quality, skew, fading, and non-standard document structure.
- Large-format architectural and engineering drawings, often image-heavy and memory-intensive during extraction.
- Current capital project documents with modern digital artifacts (PDFs, Office docs, CAD-adjacent exports, mixed metadata quality).
- Heterogeneous filename/path conventions accumulated over decades, including nested project folder layouts and inconsistent naming.

Implications for implementation agents:

- Extraction robustness matters more than elegant minimal code; fallback behavior and failure classification should be explicit.
- Memory-aware processing is essential because drawing-heavy files can spike RAM usage unpredictably.
- Idempotent retries and accurate failure semantics are critical because poor OCR/legacy scans can cause intermittent extractor failures.
- Throughput tuning must assume mixed workloads (tiny text docs + very large drawings) rather than homogeneous files.
- Logging should capture enough context (file type/stage/error class) to support operational triage at scale.

Vision across projects:

- `archives_scraper` = production baseline architecture and canonical DB behavior.
- `desktop_archives_scraper` = Windows desktop execution variant for throughput and temporary catch-up.
- `file_code_tagger` = earlier R&D codebase that provides selective implementation ideas (not architecture ownership for this repo).

Primary success condition:

- For the same input files, this desktop app writes logically equivalent records to the same DB tables as `archives_scraper` (including embeddings), with no schema changes and no Tika.

## 2) Non-Negotiable Constraints

- Heavily reuse and mirror `archives_scraper` architecture, naming, and data contract.
- Use existing production schema only; no added tables/columns/indexes in v1.
- CLI-driven operation (operators run one or more instances manually/scheduled on desktops).
- Concurrency expected: Linux scraper + 3 to 4 desktop instances simultaneously.
- Concurrency mitigation strategy: reduce write-call frequency via batch persistence and idempotent conflicts handling.
- Date extraction/tagging is out of scope for v1.
- No Tika.

## 3) Source-of-Truth Reuse Policy

When there is conflict between source repos, prefer `archives_scraper` unless a Windows-only concern requires adaptation.

### 3.1 Reuse Priority (Highest to Lowest)

1. `archives_scraper`: architecture, worker flow, DB contract, logging, failure semantics.
2. `file_code_tagger`: narrow utility patterns only where they improve Windows execution or path handling.
3. New code in this repo: only for desktop adaptation, batching, and operational packaging.

### 3.2 Copy/Adapt/Drop Matrix

Copy-first from `archives_scraper` (near 1:1 unless noted):

- CLI and loop orchestration concepts
- Worker control flow and stage transitions
- DB models/query contract with existing production tables
- Extraction registry and extractor composition
- Embedding generation path
- Logging setup and failure-table lifecycle

Adapt for desktop behavior:

- Config defaults for Windows mount/path handling
- Runtime knobs for memory-sensitive large-format files
- Batched persistence APIs to reduce transaction churn

Drop for MVP:

- `file_code_tagger` tagging/date pipelines
- COM-specific extras not required for parity
- Any non-essential future-analysis logic

## 4) Build-Agent Instructions

This section is intended to be consumed by implementation agents.

### 4.1 Agent Mission

Build a Windows desktop scraper that is behaviorally equivalent to `archives_scraper` at the DB boundary, while improving throughput via controlled batching and reduced DB write call count.

### 4.2 Agent Guardrails

- Do not redesign pipeline architecture.
- Do not invent new DB schema.
- Do not expand scope to tagging/date extraction.
- Do not introduce Tika.
- Preserve failure semantics (`file_content_failures` lifecycle behavior).
- Keep module boundaries aligned with `archives_scraper` to ease diff/review.

### 4.3 Required Agent Output Artifacts

Agents must produce:

1. Code scaffold and executable CLI
2. Config and environment documentation
3. Reuse map documenting each major module’s source (`archives_scraper` vs `file_code_tagger` vs new)
4. Ops runbook for multi-instance desktop usage
5. Verification evidence from smoke run and concurrent run

## 5) Target Repository Structure

Use this structure unless a strong implementation reason requires minor deviation:

```text
desktop_archives_scraper/
  __init__.py
  cli.py
  config.py
  worker.py
  logging_configuration.py
  db/
   __init__.py
   db.py
   models.py
   queries.py
  text_extraction/
   ...
  embedding/
   minilm.py
docs/
  architecture.md
  configuration.md
  operations.md
  schema.md
  testing.md
README.md
```

## 6) Implementation Roadmap (Phased)

### Phase 0 — Baseline Alignment

1. Update `README.md` with project purpose, relationship to `archives_scraper`, and temporary enterprise role.
2. Define explicit scope/non-scope and parity target.
3. Document required environment variables and mounted file server assumptions.

Exit criteria:

- A new engineer can explain exactly why this repo exists and what “done” means.

### Phase 1 — Core Port (Parity First)

1. Establish package layout and CLI entrypoint.
2. Port worker loop and DB integration from `archives_scraper`.
3. Port extraction + embedding stack required for parity.
4. Ensure outputs land in same tables with same logical semantics.

Exit criteria:

- Single instance can process sample files end-to-end and populate expected DB rows.

### Phase 2 — Desktop Performance and Concurrency Hardening

1. Implement write coalescing:
  - aggregate successful content writes per batch
  - aggregate failure upserts per batch
2. Reduce commit frequency with bounded batch and optional commit interval.
3. Ensure duplicate-claim/duplicate-write conflicts are treated as benign idempotent outcomes where appropriate.
4. Validate no false failure inflation from race conditions.

Exit criteria:

- 2–4 concurrent instances produce stable outcomes with materially fewer DB write calls than per-file commits.

### Phase 3 — Operations and Reliability

1. Standardize logging fields for run diagnostics.
2. Document tuning presets for different desktop capacities.
3. Add smoke-test workflow and troubleshooting guidance.

Exit criteria:

- Operators can run and tune multi-instance scraping without reading source code.

## 7) Concurrency Strategy (No Schema Changes)

Goal: keep correctness while reducing lock pressure and write amplification.

Approach:

- Use current selection/processing model from `archives_scraper` as base.
- Add batch persistence so each loop produces fewer SQL write operations.
- Keep all writes idempotent where constraints allow (`ON CONFLICT` patterns).
- Treat known duplicate-success conflicts as non-fatal and do not convert them into failure rows.
- Preserve failure tracking for real extraction/embedding errors only.

Tuning knobs to expose:

- `poll_batch_size`
- `write_batch_size`
- `commit_interval_seconds`
- `max_workers` / local process count guidance

## 8) Detailed Task Checklist for Building Agents

1. **Scaffold and entrypoint**
  - Create package modules and wire `python -m desktop_archives_scraper.cli`.
2. **Config + env**
  - Port env loading pattern and define required vars for DB/mount paths.
3. **DB layer**
  - Port models/queries that match production schema.
  - Add bulk-write helper methods (success + failure paths).
4. **Worker**
  - Port processing loop and integrate batched write calls.
5. **Extraction/embedding**
  - Port required extractors and MiniLM embedding path from `archives_scraper`.
6. **Logging/failures**
  - Port structured logging config and failure semantics.
7. **Docs**
  - Write architecture/config/operations/schema/testing docs.
8. **Validation**
  - Run smoke test and small concurrent test; capture outcomes.

## 9) Definition of Done

All items below must be true:

- CLI runs successfully and processes mounted file paths.
- DB outputs match `archives_scraper` contract for successful and failed files.
- Embeddings are written as expected.
- No schema changes introduced.
- Multi-instance runs (desktop + Linux coexistence) complete without systemic duplicate-write failure noise.
- Write call count is reduced versus naive per-file commits.
- Documentation enables handoff to operators and future maintainers.

## 10) Verification Plan

Functional checks:

- `uv sync`
- `uv run python -m desktop_archives_scraper.cli --help`
- Single-instance smoke run on representative sample set

Parity checks:

- Compare inserted/updated rows and failure-table behavior against `archives_scraper` expectations.

Concurrency checks:

- Run 2 to 4 concurrent desktop instances (and optionally Linux scraper concurrently).
- Confirm benign handling of duplicate claims/writes.
- Confirm failure rows represent real processing errors only.

Performance checks:

- Measure transaction/write-call volume before and after batching.
- Validate throughput and memory stability with large-format drawings.

## 11) Immediate Next Build Step

Start Phase 1 by porting minimal runnable modules from `archives_scraper` (CLI, worker, DB core, logging, extraction, embedding), then run a small end-to-end smoke test before any optimization beyond batched writes.

## 12) Research Reference Index (Local, Ignored)

The following files are copied locally under `research/reference/` for quick agent lookup and architecture alignment.

### 12.1 `archives_scraper` references

- `research/reference/archives_scraper/db/models.py`
  - Canonical table/model mapping and fields expected by production DB behavior.
- `research/reference/archives_scraper/db/db.py`
  - Session/engine/query patterns and persistence behavior baseline.
- `research/reference/archives_scraper/core/worker.py`
  - Canonical worker loop, batching cadence, and failure-handling flow.
- `research/reference/archives_scraper/core/cli.py`
  - CLI entrypoint semantics and runtime option patterns.
- `research/reference/archives_scraper/core/logging_configuration.py`
  - Logging structure and operational signal conventions.
- `research/reference/archives_scraper/development/failure_table_integration_spec.md`
  - Failure lifecycle expectations (`file_content_failures`) and retry semantics.
- `research/reference/archives_scraper/development/worker_cli_logging_spec.md`
  - Intended worker/CLI/logging behavior contract.

### 12.2 `file_code_tagger` references (selective)

- `research/reference/file_code_tagger/db/models.py`
  - Supplemental schema/history context from earlier R&D project.
- `research/reference/file_code_tagger/db/db.py`
  - Alternate DB utility patterns that may inform adapters.
- `research/reference/file_code_tagger/cli/add_files.py`
  - Legacy CLI flow for comparison only.
- `research/reference/file_code_tagger/pipeline/add_files_pipeline.py`
  - Prior pipeline sequencing ideas; not architecture source-of-truth.
- `research/reference/file_code_tagger/pipeline/date_mentions_pipeline.py`
  - Future-looking date extraction context (out of MVP scope).
- `research/reference/file_code_tagger/text_extraction/extraction_utils.py`
  - Utility patterns for extraction/path handling where relevant.

Usage rule for agents:

- Treat `archives_scraper` references as primary authority.
- Use `file_code_tagger` references only when they improve desktop adaptation without violating MVP scope.

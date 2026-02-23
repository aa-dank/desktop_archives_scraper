# worker.py

"""
Pure execution engine for file extraction and embedding.

This module provides the core worker logic for:
- Fetching unprocessed files from the database
- Extracting text using registered extractors
- Embedding text using configured embedders
- Persisting results with proper failure handling

No CLI parsing or global state - callable from anywhere.
"""
import os
import time
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from desktop_archives_scraper.text_extraction.extraction_utils import (
    common_char_replacements,
    strip_diacritics,
    normalize_unicode,
    normalize_whitespace
)

import numpy as np
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.sql import func

from desktop_archives_scraper.db.models import File, FileContent, FileContentFailure
from desktop_archives_scraper.db.queries import failure_row, persist_processing_batch
import logging

logger = logging.getLogger(__name__)

# Stage constants for file_content_failures
STAGE_EXTRACT = "extract"
STAGE_EMBED = "embed"

def assemble_file_server_filepath(base_mount: str,
                                  server_dir: str,
                                  filename: str = None) -> Path:
    r"""
    Join a server-relative path + filename onto a machine-specific
    mount-point.

    Parameters
    ----------
    base_mount : str
        The local mount of the records share, e.g.
        r"N:\PPDO\Records"  (Windows)  or  "/mnt/n/PPDO/Records" (Linux).
    server_dir : str
        The value from file_locations.file_server_directories
        (always stored with forward-slashes).
    filename   : str
        file_locations.filename

    Returns
    -------
    pathlib.Path  – ready for open(), exists(), etc.
    """
    # 1) Treat the DB field as a *POSIX* path (it always uses “/”)
    rel_parts = PurePosixPath(server_dir).parts     # -> tuple of segments

    # 2) Let Path figure out the separator style of this machine
    full_path = Path(base_mount).joinpath(*rel_parts)
    if filename:
        full_path = full_path / filename
    
    return full_path

def utcnow() -> datetime:
    """Return current UTC datetime with timezone info."""
    return datetime.now(timezone.utc)


def format_failure_error(
    *,
    raw_error: str,
    source_path: Path | None,
    server_dir: str | None,
    filename: str | None,
    temp_path: str | None = None,
) -> str:
    """
    Build a stable, useful failure message for persistence.

    Replaces temp-file paths with source path and appends source location
    context from file_locations.
    """
    message = raw_error or "unknown error"

    if temp_path and source_path:
        message = message.replace(temp_path, str(source_path))

        temp_dir = os.path.dirname(temp_path)
        if temp_dir:
            message = message.replace(temp_dir, str(source_path.parent))

    context_parts = []
    if source_path:
        context_parts.append(f"source_path={source_path}")
    if server_dir:
        context_parts.append(f"server_dir={server_dir}")
    if filename:
        context_parts.append(f"filename={filename}")

    if context_parts:
        message = f"{message} ({', '.join(context_parts)})"

    return message


def build_extractor_registry(extractors: list) -> dict[str, Any]:
    """
    Build a mapping of file extension to extractor.
    
    Extensions are normalized to lowercase without leading dots.
    Conflict resolution is last-wins (last extractor in list takes precedence).
    
    Parameters
    ----------
    extractors : list
        List of extractor instances, each with a `file_extensions` attribute.
    
    Returns
    -------
    dict[str, Any]
        Mapping of normalized extension (e.g., "pdf") to extractor instance.
    """
    registry = {}
    for extractor in extractors:
        if not hasattr(extractor, 'file_extensions'):
            logger.warning(f"Extractor {extractor} missing file_extensions attribute, skipping")
            continue
        
        for ext in extractor.file_extensions:
            # Normalize: lowercase, strip leading dot
            normalized = ext.lower().lstrip('.')
            registry[normalized] = extractor
            logger.debug(f"Registered extractor {extractor.__class__.__name__} for extension '{normalized}'")
    
    logger.info(f"Built extractor registry with {len(registry)} extensions")
    return registry


def next_files_needing_content(
    session: Session,
    *,
    extensions: set[str] | None = None,
    limit: int = 10,
    include_failures: bool = False,
    randomize: bool = False,
) -> list:
    """
    Fetch the next batch of files needing content extraction.
    
    Returns files that have no FileContent row (Option A semantics).
    By default, excludes files with existing failure records to prevent
    infinite requeue loops.
    
    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    extensions : set[str] | None
        If provided, only return files with these extensions (case-insensitive).
    limit : int, default=10
        Maximum number of files to return.
    include_failures : bool, default=False
        If True, include files that have failure records (for retry).
        If False (default), exclude files with any failure record.
    randomize : bool, default=False
        If True, randomize file order before applying limit.
    
    Returns
    -------
    list
        List of File records needing processing.
    """
    query = (
        session.query(File)
        .options(selectinload(File.locations))
        .outerjoin(FileContent, FileContent.file_hash == File.hash)
        .outerjoin(FileContentFailure, FileContentFailure.file_hash == File.hash)
    )
    
    # Filter by extensions if provided
    if extensions is not None:
        normalized = [ext.lower().lstrip('.') for ext in extensions]
        query = query.filter(func.lower(File.extension).in_(normalized))
    
    # Base condition: no successful FileContent row (Option A)
    query = query.filter(FileContent.file_hash.is_(None))
    
    # Apply failure filtering
    if not include_failures:
        # Exclude files that have a failure record
        query = query.filter(FileContentFailure.file_hash.is_(None))
    
    if randomize:
        query = query.order_by(func.random())
    else:
        query = query.order_by(File.id)

    query = query.limit(limit)
    
    files = query.all()
    logger.debug(f"Fetched {len(files)} files needing content extraction (include_failures={include_failures})")
    return files


def process_one_file(
    *,
    extractors_by_ext: dict[str, Any],
    embedder: Any,
    file_record: Any,
    now_fn: Callable[[], datetime] = utcnow,
    max_chars: int | None = None,
    enable_embedding: bool = True,
    dry_run: bool = False,
) -> dict:
    """
    Process a single file: extract text, embed, and persist results.
    
    Returns a processing payload that can be persisted in a later batch.
    
    Parameters
    ----------
    extractors_by_ext : dict[str, Any]
        Mapping from extension to extractor instance.
    embedder : Any
        Embedder instance with encode(Sequence[str]) -> list[np.ndarray] method.
    file_record : Any
        File model instance to process.
    now_fn : Callable[[], datetime]
        Function returning current UTC datetime (for testing).
    max_chars : int | None
        If set, skip files with extracted text exceeding this length.
    enable_embedding : bool
        Whether to generate embeddings (default True).
    
    Returns
    -------
    dict
        Status information with keys: status, chars, duration_ms
    dry_run : bool
        If True, payloads are returned but caller may skip persistence.
        Status values: "ok", "no_extractor", "error"
    """
    start_time = time.time()
    result = {
        "status": "error",
        "chars": 0,
        "duration_ms": 0,
    }
    current_stage = STAGE_EXTRACT
    file_path: Path | None = None
    record_location_directories: str | None = None
    record_filename: str | None = None
    temp_fp: str | None = None
    
    try:
        # Determine extension
        ext = (file_record.extension or "").lower().lstrip('.')
        
        # Select extractor
        extractor = extractors_by_ext.get(ext)
        if not extractor:
            error_msg = f"no extractor for ext={ext}"
            logger.warning(
                f"No extractor for file",
                extra={
                    "file_id": file_record.id,
                    "ext": ext,
                    "path": getattr(file_record, 'path', None),
                    "stage": STAGE_EXTRACT,
                }
            )
            result["status"] = "no_extractor"
            result["failure"] = failure_row(
                file_hash=file_record.hash,
                stage=STAGE_EXTRACT,
                error=error_msg,
                failed_at=now_fn(),
            )
            result["duration_ms"] = int((time.time() - start_time) * 1000)
            return result
        
        # Get file path from first location
        if file_record.locations:
            record_location_directories = file_record.locations[0].file_server_directories
            record_filename = file_record.locations[0].filename
            file_path = assemble_file_server_filepath(
                base_mount=os.environ.get("FILE_SERVER_MOUNT", ""),
                server_dir=record_location_directories,
                filename=record_filename,
            )
        
        if not file_path:
            error_msg = "no path available"
            logger.error(
                f"No path available for file",
                extra={"file_id": file_record.id, "stage": STAGE_EXTRACT}
            )
            result["status"] = "error"
            result["failure"] = failure_row(
                file_hash=file_record.hash,
                stage=STAGE_EXTRACT,
                error=error_msg,
                failed_at=now_fn(),
            )
            result["duration_ms"] = int((time.time() - start_time) * 1000)
            return result
        
        # Extract text
        logger.info(
            f"Extracting text from file",
            extra={
                "file_id": file_record.id,
                "path": str(file_path),
                "ext": ext,
            }
        )
        
        extracted_text = ""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_fp = os.path.join(temp_dir, os.path.basename(str(file_path)))
            shutil.copyfile(str(file_path), temp_fp)
            extracted_text = extractor(temp_fp)

        if extracted_text:
            extracted_text = common_char_replacements(extracted_text)
            extracted_text = strip_diacritics(extracted_text)
            extracted_text = normalize_unicode(extracted_text)
            extracted_text = normalize_whitespace(extracted_text)
        
        # Check text length limit
        if max_chars and len(extracted_text) > max_chars:
            error_msg = f"Extracted text length {len(extracted_text)} exceeds limit {max_chars}"
            logger.warning(
                f"Skipping file due to text length",
                extra={
                    "file_id": file_record.id,
                    "chars": len(extracted_text),
                    "limit": max_chars,
                }
            )
            
            result["status"] = "error" 
            result["failure"] = failure_row(
                file_hash=file_record.hash,
                stage=STAGE_EXTRACT,
                error=error_msg,
                failed_at=now_fn(),
            )
            result["chars"] = len(extracted_text)
            result["duration_ms"] = int((time.time() - start_time) * 1000)
            return result
        
        result["chars"] = len(extracted_text)
        
        # Generate embedding if enabled
        current_stage = STAGE_EMBED
        embedding_vector = None
        if enable_embedding and extracted_text.strip():
            logger.debug(
                f"Generating embedding for file",
                extra={"file_id": file_record.id, "chars": len(extracted_text)}
            )
            embeddings = embedder.encode([extracted_text])
            if embeddings and len(embeddings) > 0:
                embedding_vector = embeddings[0]
        
        # Build FileContent payload for batched upsert
        content_row = {
            "file_hash": file_record.hash,
            "source_text": extracted_text,
            "text_length": len(extracted_text),
            "updated_at": now_fn(),
            "minilm_model": None,
            "minilm_emb": None,
            "mpnet_model": None,
            "mpnet_emb": None,
        }
        
        if embedding_vector is not None:
            # Determine embedder model name
            model_name = getattr(embedder, 'model_name', 'unknown')
            
            # Map to appropriate column based on dimension
            if hasattr(embedding_vector, 'shape'):
                dim = embedding_vector.shape[0] if len(embedding_vector.shape) > 0 else len(embedding_vector)
            else:
                dim = len(embedding_vector)
            
            if dim == 384:  # MiniLM dimension
                content_row["minilm_emb"] = embedding_vector
                content_row["minilm_model"] = model_name
            elif dim == 768:  # MPNet dimension
                content_row["mpnet_emb"] = embedding_vector
                content_row["mpnet_model"] = model_name
            else:
                logger.warning(
                    f"Unknown embedding dimension",
                    extra={"file_id": file_record.id, "dimension": dim}
                )
        result["content"] = content_row
        
        logger.info(
            f"Successfully processed file",
            extra={
                "file_id": file_record.id,
                "chars": len(extracted_text),
                "status": "ok",
            }
        )
        
        result["status"] = "ok"
        result["duration_ms"] = int((time.time() - start_time) * 1000)
        return result
        
    except Exception as e:
        logger.exception(
            f"Error processing file",
            extra={
                "file_id": file_record.id,
                "error": str(e),
                "stage": current_stage,
            }
        )
        
        formatted_error = format_failure_error(
            raw_error=str(e),
            source_path=file_path,
            server_dir=record_location_directories,
            filename=record_filename,
            temp_path=temp_fp,
        )
        result["failure"] = failure_row(
            file_hash=file_record.hash,
            stage=current_stage,
            error=formatted_error[:500],
            failed_at=now_fn(),
        )
        
        result["status"] = "error"
        result["duration_ms"] = int((time.time() - start_time) * 1000)
        return result


def run_worker(
    *,
    dry_run: bool = False,
    session_factory: Callable,
    extractors: list,
    embedder: Any,
    poll_seconds: float = 5.0,
    poll_batch_size: int = 10,
    write_batch_size: int = 25,
    commit_interval_seconds: float = 5.0,
    limit: int | None = None,
    extensions: set[str] | None = None,
    max_chars: int | None = None,
    backoff_seconds: float | None = None,
    enable_embedding: bool = True,
    include_failures: bool = False,
    randomize: bool = False,
) -> int:
    """
    Main worker execution loop.
    
    Continuously fetches and processes files until stopped or no more work.
    
    Parameters
    ----------
    session_factory : Callable
        Factory function returning SQLAlchemy sessions (e.g., sessionmaker).
    extractors : list
        List of extractor instances.
    embedder : Any
        Embedder instance.
    poll_seconds : float, default=5.0
        Seconds to sleep between polling when no work found.
    limit : int | None
        Total files to process before exiting. If None, run continuously.
    extensions : set[str] | None
        If provided, only process files with these extensions.
    max_chars : int | None
        If set, skip files with extracted text exceeding this length.
    backoff_seconds : float | None, default=None
        Seconds to sleep when no work is found before next poll. If None,
        uses poll_seconds.
    enable_embedding : bool, default=True
        Whether to generate embeddings.
    include_failures : bool, default=False
        If True, include files with failure records for retry.
        If False (default), exclude files that have previously failed.
    randomize : bool, default=False
        If True, randomize file retrieval order each batch.
    
    Returns
    -------
    int
        Exit code: 0 (clean), 2 (config error), 3 (runtime failure)
    """
    # Validate configuration
    if not extractors:
        logger.error("No extractors provided")
        return 2
    
    if enable_embedding and not embedder:
        logger.error("Embedding enabled but no embedder provided")
        return 2

    if poll_batch_size < 1:
        logger.error("poll_batch_size must be >= 1")
        return 2

    if write_batch_size < 1:
        logger.error("write_batch_size must be >= 1")
        return 2
    
    # Build extractor registry
    registry = build_extractor_registry(extractors)
    if not registry:
        logger.error("Failed to build extractor registry")
        return 2

    # Restrict extensions to those supported by extractors
    supported_extensions = set(registry.keys())
    if extensions:
        extensions = extensions.intersection(supported_extensions)
    else:
        extensions = supported_extensions
    
    logger.info(
        f"Worker starting",
        extra={
            "poll_seconds": poll_seconds,
            "poll_batch_size": poll_batch_size,
            "write_batch_size": write_batch_size,
            "commit_interval_seconds": commit_interval_seconds,
            "limit": limit,
            "extensions": list(extensions) if extensions else None,
            "enable_embedding": enable_embedding,
            "include_failures": include_failures,
            "randomize": randomize,
        }
    )
    
    total_processed = 0
    pending_content_rows: list[dict] = []
    pending_failure_rows: list[dict] = []
    last_flush_ts = time.time()

    idle_sleep_seconds = poll_seconds if backoff_seconds is None else backoff_seconds

    def flush_pending(force: bool = False) -> None:
        nonlocal pending_content_rows, pending_failure_rows, last_flush_ts

        has_pending = bool(pending_content_rows or pending_failure_rows)
        if not has_pending:
            return

        now_ts = time.time()
        interval_elapsed = (now_ts - last_flush_ts) >= commit_interval_seconds
        size_reached = (len(pending_content_rows) + len(pending_failure_rows)) >= write_batch_size

        if not force and not interval_elapsed and not size_reached:
            return

        with session_factory() as flush_session:
            content_count, failure_count, cleared_count = persist_processing_batch(
                flush_session,
                content_rows=pending_content_rows,
                failure_rows=pending_failure_rows,
            )

        logger.info(
            "Flushed pending writes",
            extra={
                "content_upserts": content_count,
                "failure_upserts": failure_count,
                "failure_rows_cleared": cleared_count,
            },
        )

        pending_content_rows = []
        pending_failure_rows = []
        last_flush_ts = now_ts
    
    try:
        while True:
            with session_factory() as session:
                # Fetch next batch
                remaining = None
                if limit is not None:
                    remaining = max(limit - total_processed, 0)
                batch_limit = poll_batch_size if remaining is None else min(poll_batch_size, remaining)

                if batch_limit == 0:
                    if not dry_run:
                        flush_pending(force=True)
                    logger.info(f"Reached limit, processed {total_processed} files")
                    return 0

                files = next_files_needing_content(
                    session,
                    extensions=extensions,
                    limit=batch_limit,
                    include_failures=include_failures,
                    randomize=randomize,
                )
                
                if not files:
                    if not dry_run:
                        flush_pending(force=True)
                    logger.info("No files needing processing")
                    if limit is not None:
                        logger.info(f"Exiting after processing {total_processed} files (limit reached or no work)")
                        return 0
                    
                    logger.debug(f"Sleeping {idle_sleep_seconds}s before next poll")
                    time.sleep(idle_sleep_seconds)
                    continue
                
                logger.info(f"Processing batch of {len(files)} files")
                
                # Process each file
                batch_results = {"ok": 0, "no_extractor": 0, "error": 0}
                for file_record in files:
                    result = process_one_file(
                        extractors_by_ext=registry,
                        embedder=embedder,
                        file_record=file_record,
                        max_chars=max_chars,
                        enable_embedding=enable_embedding,
                        dry_run=dry_run,
                    )
                    batch_results[result["status"]] += 1
                    total_processed += 1

                    if not dry_run:
                        content_row = result.get("content")
                        failure = result.get("failure")
                        if content_row is not None:
                            pending_content_rows.append(content_row)
                        if failure is not None:
                            pending_failure_rows.append(failure)
                        flush_pending(force=False)
                
                logger.info(
                    f"Batch complete",
                    extra={
                        "processed": len(files),
                        "ok": batch_results["ok"],
                        "no_extractor": batch_results["no_extractor"],
                        "errors": batch_results["error"],
                    }
                )
                
                if limit is not None and total_processed >= limit:
                    if not dry_run:
                        flush_pending(force=True)
                    logger.info(f"Reached limit, processed {total_processed} files")
                    return 0
                
                # Brief sleep before next batch
                if poll_seconds > 0:
                    time.sleep(poll_seconds)
    
    except KeyboardInterrupt:
        if not dry_run:
            flush_pending(force=True)
        logger.info(f"Worker interrupted, processed {total_processed} files")
        return 0
    
    except Exception as e:
        if not dry_run:
            try:
                flush_pending(force=True)
            except Exception:
                logger.exception("Failed to flush pending writes during fatal shutdown")
        logger.exception(f"Unexpected worker failure: {e}")
        return 3

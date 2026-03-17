# cli.py

"""
Command-line interface for the extraction worker.

This module provides a thin operational wrapper around run_worker() with
no business logic. Supports both CLI arguments and environment variables
for AWS/headless deployment.
"""

import os
import sys

import click
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from desktop_archives_scraper.config import load_worker_config
from desktop_archives_scraper.db.db import get_db_engine
from desktop_archives_scraper.logging_configuration import configure_logging, get_logger
from desktop_archives_scraper.text_extraction.basic_extraction import TextFileTextExtractor
from desktop_archives_scraper.text_extraction.image_extraction import ImageTextExtractor
from desktop_archives_scraper.text_extraction.office_doc_extraction import (
    PresentationTextExtractor,
    SpreadsheetTextExtractor,
    WordFileTextExtractor,
)
from desktop_archives_scraper.text_extraction.pdf_extraction import PDFTextExtractor
from desktop_archives_scraper.text_extraction.web_extraction import EmailTextExtractor, HtmlTextExtractor
from desktop_archives_scraper.worker import run_worker


# Load environment variables early so Click options using envvar=... can
# see values from a local .env file.
load_dotenv(override=False)


@click.command()
@click.option(
    "--limit",
    type=int,
    envvar="LIMIT",
    help="Total files to process before exiting. If omitted, run continuously.",
)
@click.option(
    "--poll-seconds",
    type=float,
    envvar="POLL_SECONDS",
    show_default=False,
    help="Seconds between batch polls",
)
@click.option(
    "--poll-batch-size",
    type=int,
    envvar="POLL_BATCH_SIZE",
    show_default=False,
    help="Max files fetched per polling query",
)
@click.option(
    "--write-batch-size",
    type=int,
    envvar="WRITE_BATCH_SIZE",
    show_default=False,
    help="Max processed files queued before a DB batch flush",
)
@click.option(
    "--commit-interval-seconds",
    type=float,
    envvar="COMMIT_INTERVAL_SECONDS",
    show_default=False,
    help="Time-based flush interval for pending DB writes",
)
@click.option(
    "--extensions",
    type=str,
    envvar="EXTENSIONS",
    help="Comma-separated file extensions to process",
)
@click.option(
    "--max-chars",
    type=int,
    envvar="MAX_CHARS",
    help="Maximum characters to extract. Files exceeding this limit will be recorded as failures and skipped.",
)
@click.option(
    "--embed/--no-embed",
    "enable_embedding",
    default=True,
    envvar="ENABLE_EMBEDDING",
    show_default=True,
    help="Enable/disable embedding generation",
)
@click.option(
    "--date-extract/--no-date-extract",
    "enable_date_extraction",
    default=True,
    show_default=True,
    help="Enable/disable date mention extraction",
)
@click.option(
    "--embedder",
    type=click.Choice(["minilm"], case_sensitive=False),
    default="minilm",
    envvar="EMBEDDER",
    show_default=True,
    help="Embedder model to use",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    envvar="LOG_LEVEL",
    show_default=True,
    help="Logging level",
)
@click.option(
    "--log-file",
    type=click.Path(),
    envvar="LOG_FILE",
    help="Path to log file",
)
@click.option(
    "--json-logs",
    is_flag=True,
    envvar="JSON_LOGS",
    help="Output logs in JSON format",
)
@click.option(
    "--max-runtime-seconds",
    type=float,
    envvar="MAX_RUNTIME_SECONDS",
    help="Maximum runtime in seconds before exiting cleanly.",
)
@click.option(
    "--failure_retry_treshold",
    type=click.IntRange(min=1),
    envvar="FAILURE_RETRY_TRESHOLD",
    default=None,
    help="Retry files with previous failures only when failure attempts are below this value",
)
@click.option(
    "--randomize/--no-randomize",
    "randomize",
    default=False,
    envvar="RANDOMIZE",
    show_default=True,
    help="Randomize database file selection order for scraping",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Perform dry run without persisting changes",
)
def main(
    limit: int | None,
    poll_seconds: float | None,
    poll_batch_size: int | None,
    write_batch_size: int | None,
    commit_interval_seconds: float | None,
    extensions: str | None,
    max_chars: int | None,
    enable_embedding: bool,
    enable_date_extraction: bool,
    embedder: str,
    log_level: str,
    log_file: str | None,
    json_logs: bool,
    max_runtime_seconds: float | None,
    failure_retry_treshold: int | None,
    randomize: bool,
    dry_run: bool,
) -> None:
    """
    File extraction and embedding worker.
    
    Processes files from the database, extracts text, generates embeddings,
    and persists results. Runs continuously unless a limit is provided.
    """
    # Configure logging first
    configure_logging(
        level=log_level,
        log_file=log_file,
        console=True,
        json_format=json_logs,
    )
    
    logger = get_logger(__name__)
    defaults = load_worker_config()

    resolved_poll_seconds = defaults.poll_seconds if poll_seconds is None else poll_seconds
    resolved_poll_batch_size = defaults.poll_batch_size if poll_batch_size is None else max(poll_batch_size, 1)
    resolved_write_batch_size = defaults.write_batch_size if write_batch_size is None else max(write_batch_size, 1)
    resolved_commit_interval = (
        defaults.commit_interval_seconds
        if commit_interval_seconds is None
        else max(commit_interval_seconds, 0.0)
    )

    logger.info(
        "Runtime tuning",
        extra={
            "poll_seconds": resolved_poll_seconds,
            "poll_batch_size": resolved_poll_batch_size,
            "write_batch_size": resolved_write_batch_size,
            "commit_interval_seconds": resolved_commit_interval,
        },
    )
    
    logger.info("Starting extraction worker CLI")
    
    if dry_run:
        logger.info("Dry run enabled: no database changes will be persisted")
    
    # Parse extensions
    ext_set = None
    if extensions:
        ext_set = set(ext.strip() for ext in extensions.split(",") if ext.strip())

    logger.info(
        "Extraction settings",
        extra={
            "extensions": sorted(ext_set) if ext_set else "all",
            "max_chars": max_chars,
            "enable_date_extraction": enable_date_extraction,
            "max_runtime_seconds": max_runtime_seconds,
            "limit": limit,
            "randomize": randomize,
            "failure_retry_treshold": failure_retry_treshold,
            "dry_run": dry_run,
        },
    )

    logger.info(
        "Embedding settings",
        extra={
            "enable_embedding": enable_embedding,
            "embedder": embedder if enable_embedding else None,
        },
    )

    # Build extractors
    extractors = [
        PDFTextExtractor(),
        ImageTextExtractor(),
        WordFileTextExtractor(),
        PresentationTextExtractor(),
        SpreadsheetTextExtractor(),
        HtmlTextExtractor(),
        EmailTextExtractor(),
        TextFileTextExtractor(),
    ]
    logger.info(f"Initialized {len(extractors)} extractors")
    
    # Build embedder (lazy import to avoid loading heavy dependencies on --help)
    embedder_instance = None
    if enable_embedding:
        if embedder == "minilm":
            from desktop_archives_scraper.embedding.minilm import MiniLMEmbedder
            embedder_instance = MiniLMEmbedder()
            logger.info(f"Initialized MiniLM embedder (dim={embedder_instance.dim})")
        else:
            logger.error(f"Unknown embedder: {embedder}")
            sys.exit(2)
    else:
        logger.info("Embedding disabled")
    
    # Create database session factory — probe connectivity before any heavy work
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        session_factory = sessionmaker(bind=engine)
        logger.info("Database connection established")
    
    # Handle common database connection errors with user-friendly messages
    except OperationalError as e:
        err_msg = str(e.orig).lower() if e.orig else str(e).lower()
        if "password authentication failed" in err_msg:
            logger.error(
                "Database authentication failed: incorrect username or password.",
                extra={"db_host": os.getenv("DB_HOST"), "db_user": os.getenv("DB_USERNAME")},
            )
        elif "database" in err_msg and "does not exist" in err_msg:
            logger.error(
                f"Database '{os.getenv('DB_NAME')}' does not exist on the server.",
                extra={"db_host": os.getenv("DB_HOST"), "db_name": os.getenv("DB_NAME")},
            )
        elif "role" in err_msg and "does not exist" in err_msg:
            logger.error(
                f"Database role/user '{os.getenv('DB_USERNAME')}' does not exist.",
                extra={"db_host": os.getenv("DB_HOST"), "db_user": os.getenv("DB_USERNAME")},
            )
        elif any(phrase in err_msg for phrase in ("could not connect", "connection refused", "no route to host")):
            logger.error(
                "Cannot reach the database server — check that the host and port are correct and the server is reachable.",
                extra={"db_host": os.getenv("DB_HOST"), "db_port": os.getenv("DB_PORT")},
            )
        elif "timeout" in err_msg or "timed out" in err_msg:
            logger.error(
                "Connection to the database timed out — the host may be unreachable or behind a firewall.",
                extra={"db_host": os.getenv("DB_HOST"), "db_port": os.getenv("DB_PORT")},
            )
        elif "name or service not known" in err_msg or "could not translate host" in err_msg:
            logger.error(
                f"Cannot resolve database hostname '{os.getenv('DB_HOST')}' — check DNS or the DB_HOST value.",
                extra={"db_host": os.getenv("DB_HOST")},
            )
        else:
            logger.error(f"Failed to connect to database: {e}")
        sys.exit(2)
    except Exception as e:
        logger.error(f"Unexpected error while connecting to database: {e}")
        sys.exit(2)
    
    # Run worker
    try:
        exit_code = run_worker(
            session_factory=session_factory,
            extractors=extractors,
            embedder=embedder_instance,
            poll_seconds=resolved_poll_seconds,
            poll_batch_size=resolved_poll_batch_size,
            write_batch_size=resolved_write_batch_size,
            commit_interval_seconds=resolved_commit_interval,
            limit=limit,
            max_runtime_seconds=max_runtime_seconds,
            extensions=ext_set,
            max_chars=max_chars,
            enable_embedding=enable_embedding,
            enable_date_extraction=enable_date_extraction,
            failure_retry_treshold=failure_retry_treshold,
            randomize=randomize,
            dry_run=dry_run,
        )
        
        logger.info(f"Worker exited with code {exit_code}")
        sys.exit(exit_code)
        
    except Exception as e:
        logger.exception(f"Fatal error in worker: {e}")
        sys.exit(3)


if __name__ == "__main__":
    sys.exit(main())

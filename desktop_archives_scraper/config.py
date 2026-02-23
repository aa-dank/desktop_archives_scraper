from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class WorkerConfig:
	poll_seconds: float = 5.0
	poll_batch_size: int = 10
	write_batch_size: int = 25
	commit_interval_seconds: float = 5.0


def _get_env_float(name: str, default: float) -> float:
	value = os.getenv(name)
	if value is None:
		return default
	try:
		return float(value)
	except ValueError:
		return default


def _get_env_int(name: str, default: int) -> int:
	value = os.getenv(name)
	if value is None:
		return default
	try:
		return int(value)
	except ValueError:
		return default


def load_worker_config() -> WorkerConfig:
	return WorkerConfig(
		poll_seconds=_get_env_float("POLL_SECONDS", 5.0),
		poll_batch_size=max(_get_env_int("POLL_BATCH_SIZE", 10), 1),
		write_batch_size=max(_get_env_int("WRITE_BATCH_SIZE", 25), 1),
		commit_interval_seconds=max(_get_env_float("COMMIT_INTERVAL_SECONDS", 5.0), 0.0),
	)

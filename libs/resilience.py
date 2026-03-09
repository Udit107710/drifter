"""Shared resilience utilities: error classification, transient detection, and retry."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def is_transient_error(exc: Exception) -> bool:
    """Return True if *exc* is likely transient and worth retrying.

    Covers stdlib network errors and common httpx timeout/connection errors.
    """
    # Stdlib transient errors
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True

    # httpx transient errors (check by class name to avoid hard dependency)
    return type(exc).__name__ in (
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "ConnectError",
        "RemoteProtocolError",
    )


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for retry with exponential backoff."""

    max_retries: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 30.0
    jitter_factor: float = 0.5


@dataclass(frozen=True)
class RetryOutcome(Generic[T]):
    """Result of a resilient_call attempt."""

    value: T | None
    exception: Exception | None
    attempts: int
    total_delay_s: float
    succeeded: bool


def _compute_delay(
    attempt: int, config: RetryConfig,
) -> float:
    """Compute delay with exponential backoff and jitter."""
    base = min(config.base_delay_s * (2 ** attempt), config.max_delay_s)
    jitter = random.uniform(0, config.jitter_factor * base)
    return base + jitter


def resilient_call(
    fn: Callable[[], T],
    config: RetryConfig,
    is_retryable: Callable[[Exception], bool] = is_transient_error,
    *,
    _sleep: Callable[[float], None] = time.sleep,
) -> RetryOutcome[T]:
    """Call *fn* with retry and exponential backoff.

    Args:
        fn: Zero-argument callable to invoke.
        config: Retry configuration.
        is_retryable: Predicate to decide if an exception is worth retrying.
        _sleep: Injectable sleep function (for testing).

    Returns:
        RetryOutcome with value on success, or last exception on exhaustion.
    """
    total_delay = 0.0
    last_exc: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            value = fn()
            return RetryOutcome(
                value=value,
                exception=None,
                attempts=attempt + 1,
                total_delay_s=total_delay,
                succeeded=True,
            )
        except Exception as exc:
            last_exc = exc

            if not is_retryable(exc):
                logger.debug(
                    "resilient_call: permanent error on attempt %d: %s",
                    attempt + 1, exc,
                )
                return RetryOutcome(
                    value=None,
                    exception=exc,
                    attempts=attempt + 1,
                    total_delay_s=total_delay,
                    succeeded=False,
                )

            if attempt < config.max_retries:
                delay = _compute_delay(attempt, config)
                logger.debug(
                    "resilient_call: transient error on attempt %d, "
                    "retrying in %.3fs: %s",
                    attempt + 1, delay, exc,
                )
                _sleep(delay)
                total_delay += delay

    return RetryOutcome(
        value=None,
        exception=last_exc,
        attempts=config.max_retries + 1,
        total_delay_s=total_delay,
        succeeded=False,
    )


async def async_resilient_call(
    fn: Callable[[], T],
    config: RetryConfig,
    is_retryable: Callable[[Exception], bool] = is_transient_error,
) -> RetryOutcome[T]:
    """Async version of resilient_call using asyncio.sleep for backoff."""
    total_delay = 0.0
    last_exc: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            value = fn()
            return RetryOutcome(
                value=value,
                exception=None,
                attempts=attempt + 1,
                total_delay_s=total_delay,
                succeeded=True,
            )
        except Exception as exc:
            last_exc = exc

            if not is_retryable(exc):
                return RetryOutcome(
                    value=None,
                    exception=exc,
                    attempts=attempt + 1,
                    total_delay_s=total_delay,
                    succeeded=False,
                )

            if attempt < config.max_retries:
                delay = _compute_delay(attempt, config)
                logger.debug(
                    "async_resilient_call: transient error on attempt %d, "
                    "retrying in %.3fs: %s",
                    attempt + 1, delay, exc,
                )
                await asyncio.sleep(delay)
                total_delay += delay

    return RetryOutcome(
        value=None,
        exception=last_exc,
        attempts=config.max_retries + 1,
        total_delay_s=total_delay,
        succeeded=False,
    )

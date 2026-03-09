"""Tests for libs/resilience.py — RetryConfig, RetryOutcome, resilient_call."""

from __future__ import annotations

from libs.resilience import RetryConfig, RetryOutcome, resilient_call


class TestResilientCallSuccess:
    def test_immediate_success(self) -> None:
        result = resilient_call(lambda: 42, RetryConfig())
        assert result.succeeded is True
        assert result.value == 42
        assert result.attempts == 1
        assert result.total_delay_s == 0.0
        assert result.exception is None


class TestResilientCallTransient:
    def test_transient_then_success(self) -> None:
        """Transient error on first attempt, success on second."""
        call_count = 0

        def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("transient")
            return "ok"

        delays: list[float] = []
        result = resilient_call(
            fn, RetryConfig(max_retries=3), _sleep=delays.append,
        )
        assert result.succeeded is True
        assert result.value == "ok"
        assert result.attempts == 2
        assert len(delays) == 1

    def test_exhausted(self) -> None:
        """All retries exhausted → failed."""
        def fn() -> str:
            raise ConnectionError("always fails")

        delays: list[float] = []
        config = RetryConfig(max_retries=2, base_delay_s=0.1)
        result = resilient_call(fn, config, _sleep=delays.append)
        assert result.succeeded is False
        assert result.attempts == 3  # initial + 2 retries
        assert isinstance(result.exception, ConnectionError)
        assert len(delays) == 2


class TestResilientCallPermanent:
    def test_permanent_error_no_retry(self) -> None:
        """Non-retryable error → immediate failure, no retries."""
        def fn() -> str:
            raise ValueError("permanent")

        delays: list[float] = []
        result = resilient_call(fn, RetryConfig(max_retries=3), _sleep=delays.append)
        assert result.succeeded is False
        assert result.attempts == 1
        assert isinstance(result.exception, ValueError)
        assert len(delays) == 0


class TestBackoffDelays:
    def test_exponential_growth(self) -> None:
        """Delays should grow exponentially (base * 2^attempt)."""
        def fn() -> str:
            raise ConnectionError("fail")

        delays: list[float] = []
        config = RetryConfig(
            max_retries=3, base_delay_s=1.0, max_delay_s=100.0, jitter_factor=0.0,
        )
        resilient_call(fn, config, _sleep=delays.append)
        # With jitter_factor=0: delays should be 1.0, 2.0, 4.0
        assert delays == [1.0, 2.0, 4.0]

    def test_max_delay_cap(self) -> None:
        """Delay should be capped at max_delay_s."""
        def fn() -> str:
            raise ConnectionError("fail")

        delays: list[float] = []
        config = RetryConfig(
            max_retries=5, base_delay_s=1.0, max_delay_s=3.0, jitter_factor=0.0,
        )
        resilient_call(fn, config, _sleep=delays.append)
        # 1.0, 2.0, 3.0, 3.0, 3.0 (capped at 3)
        assert all(d <= 3.0 for d in delays)
        assert delays[0] == 1.0
        assert delays[1] == 2.0
        assert delays[2] == 3.0

    def test_jitter_bounds(self) -> None:
        """With jitter, delay should be in [base, base + jitter_factor*base]."""
        def fn() -> str:
            raise ConnectionError("fail")

        delays: list[float] = []
        config = RetryConfig(
            max_retries=1, base_delay_s=1.0, max_delay_s=100.0, jitter_factor=0.5,
        )
        # Run many times to check bounds
        for _ in range(50):
            delays.clear()
            resilient_call(fn, config, _sleep=delays.append)
            assert len(delays) == 1
            assert 1.0 <= delays[0] <= 1.5  # base + up to 50% jitter


class TestZeroRetries:
    def test_zero_retries_success(self) -> None:
        result = resilient_call(lambda: "ok", RetryConfig(max_retries=0))
        assert result.succeeded is True
        assert result.attempts == 1

    def test_zero_retries_failure(self) -> None:
        def fn() -> str:
            raise ConnectionError("fail")

        delays: list[float] = []
        result = resilient_call(
            fn, RetryConfig(max_retries=0), _sleep=delays.append,
        )
        assert result.succeeded is False
        assert result.attempts == 1
        assert len(delays) == 0


class TestCustomPredicate:
    def test_custom_is_retryable(self) -> None:
        """Custom predicate controls which errors are retried."""
        call_count = 0

        def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("custom retryable")
            return "ok"

        delays: list[float] = []
        result = resilient_call(
            fn,
            RetryConfig(max_retries=3),
            is_retryable=lambda e: isinstance(e, RuntimeError),
            _sleep=delays.append,
        )
        assert result.succeeded is True
        assert result.attempts == 3


class TestRetryOutcome:
    def test_frozen(self) -> None:
        outcome: RetryOutcome[int] = RetryOutcome(
            value=1, exception=None, attempts=1,
            total_delay_s=0.0, succeeded=True,
        )
        assert outcome.value == 1


class TestRetryConfig:
    def test_defaults(self) -> None:
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay_s == 0.5
        assert config.max_delay_s == 30.0
        assert config.jitter_factor == 0.5

    def test_frozen(self) -> None:
        config = RetryConfig()
        try:
            config.max_retries = 5  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except AttributeError:
            pass

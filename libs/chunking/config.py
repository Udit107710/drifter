"""Chunking configuration dataclasses with validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FixedWindowConfig:
    """Configuration for fixed-window chunking strategy."""

    chunk_size: int = 256
    overlap: int = 64
    min_chunk_size: int = 32

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.overlap < 0:
            raise ValueError("overlap must be >= 0")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be < chunk_size")
        if self.min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be > 0")
        if self.min_chunk_size > self.chunk_size:
            raise ValueError("min_chunk_size must be <= chunk_size")


@dataclass(frozen=True)
class RecursiveConfig:
    """Configuration for recursive/structural chunking strategy."""

    max_chunk_size: int = 512
    min_chunk_size: int = 64
    prefer_structural: bool = True

    def __post_init__(self) -> None:
        if self.max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be > 0")
        if self.min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be > 0")
        if self.min_chunk_size > self.max_chunk_size:
            raise ValueError("min_chunk_size must be <= max_chunk_size")


@dataclass(frozen=True)
class ParentChildConfig:
    """Configuration for parent-child chunking strategy."""

    parent_chunk_size: int = 1024
    child_chunk_size: int = 256
    child_overlap: int = 64
    min_child_size: int = 32

    def __post_init__(self) -> None:
        if self.parent_chunk_size <= self.child_chunk_size:
            raise ValueError("parent_chunk_size must be > child_chunk_size")
        if self.child_chunk_size <= 0:
            raise ValueError("child_chunk_size must be > 0")
        if self.child_overlap < 0:
            raise ValueError("child_overlap must be >= 0")
        if self.child_overlap >= self.child_chunk_size:
            raise ValueError("child_overlap must be < child_chunk_size")
        if self.min_child_size <= 0:
            raise ValueError("min_child_size must be > 0")
        if self.min_child_size > self.child_chunk_size:
            raise ValueError("min_child_size must be <= child_chunk_size")

"""GPU memory profiling helpers for training loops.

The decorators in this module are intentionally independent of the training
implementations. ``profile_gpu_memory`` records one row for a batch processor,
while ``collect_memory_usage`` owns the records for one call to ``train`` and
writes them to ``memory_usage.csv`` in that training run's output folder.
"""

from __future__ import annotations

import csv
import functools
import inspect
import os
from contextvars import ContextVar
from typing import Any, Optional

import torch


_BYTES_PER_MB = 1024**2
_MEMORY_RECORDS: ContextVar[Optional[list[dict[str, Any]]]] = ContextVar(
    "gpu_memory_records", default=None
)
_CSV_FIELDS = (
    "step",
    "process_batch",
    "device",
    "cuda_available",
    "memory_before_mb",
    "peak_memory_mb",
    "peak_memory_difference_mb",
)


def _cuda_device(value: Any) -> Optional[torch.device]:
    """Find the first CUDA tensor/device nested in an argument."""
    if isinstance(value, torch.Tensor) and value.device.type == "cuda":
        return value.device
    if isinstance(value, torch.device) and value.type == "cuda":
        return value
    if isinstance(value, dict):
        values = value.values()
    elif isinstance(value, (tuple, list)):
        values = value
    else:
        return None

    for nested_value in values:
        device = _cuda_device(nested_value)
        if device is not None:
            return device
    return None


def _device_for_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Optional[torch.device]:
    if not torch.cuda.is_available():
        return None
    return (
        _cuda_device(args)
        or _cuda_device(kwargs)
        or torch.device("cuda", torch.cuda.current_device())
    )


def profile_gpu_memory(func):
    """Record peak CUDA allocated-memory growth for one function call.

    The wrapped function's arguments, return value, and exceptions are left
    unchanged. Measurements are retained only while a function wrapped by
    :func:`collect_memory_usage` is active.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        records = _MEMORY_RECORDS.get()
        if records is None:
            return func(*args, **kwargs)

        device = _device_for_call(args, kwargs)
        if device is None:
            memory_before = 0
        else:
            # CUDA work is asynchronous. Synchronizing on both sides makes the
            # measured interval match the process_batch call.
            torch.cuda.synchronize(device)
            memory_before = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)

        try:
            return func(*args, **kwargs)
        finally:
            if device is None:
                peak_memory = 0
            else:
                torch.cuda.synchronize(device)
                peak_memory = torch.cuda.max_memory_allocated(device)

            records.append(
                {
                    "step": len(records) + 1,
                    "process_batch": func.__qualname__,
                    "device": str(device) if device is not None else "cpu",
                    "cuda_available": device is not None,
                    "memory_before_mb": memory_before / _BYTES_PER_MB,
                    "peak_memory_mb": peak_memory / _BYTES_PER_MB,
                    "peak_memory_difference_mb": max(
                        0, peak_memory - memory_before
                    )
                    / _BYTES_PER_MB,
                }
            )

    return wrapper


def _training_folder(func, args, kwargs) -> str:
    """Resolve the ``folder`` argument without changing train signatures."""
    try:
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
    except TypeError:
        bound = None
    folder = bound.arguments.get("folder") if bound is not None else None
    if folder is None:
        folder = kwargs.get("folder", ".")
    return os.fspath(folder)


def _write_memory_usage(folder: str, records: list[dict[str, Any]]) -> None:
    os.makedirs(folder, exist_ok=True)
    csv_path = os.path.join(folder, "memory_usage.csv")
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def collect_memory_usage(func):
    """Collect profiled batch calls made by ``train`` and write their CSV."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        records: list[dict[str, Any]] = []
        token = _MEMORY_RECORDS.set(records)
        try:
            return func(*args, **kwargs)
        finally:
            try:
                _write_memory_usage(_training_folder(func, args, kwargs), records)
            finally:
                _MEMORY_RECORDS.reset(token)

    return wrapper


from __future__ import annotations

import math
import os
import random
from bisect import bisect_right
from collections.abc import MutableSequence, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

T = TypeVar("T")


LOG_BUCKETS = 8
MAX_BUCKETS = 1 << LOG_BUCKETS
BASE_CASE_SIZE = 32
PARALLEL_THRESHOLD = 1_000
OVERSAMPLING_FACTOR_PERCENT = 20


def _log2_floor(n: int) -> int:
    if n <= 0:
        raise ValueError("n must be > 0")
    return n.bit_length() - 1


def _oversampling_factor(n: int) -> int:
    f = OVERSAMPLING_FACTOR_PERCENT / 100.0 * math.log2(n)
    return max(1, int(f))



def _insertion_sort(
    a: MutableSequence[T],
    lo: int,
    hi: int,
    key: Callable[[T], object] | None,
) -> None:
    if hi - lo <= 1:
        return

    if key is None:
        def get_key(x):
            return x
    else:
        get_key = key

    for i in range(lo + 1, hi):
        v = a[i]
        kv = get_key(v)
        j = i - 1
        while j >= lo and get_key(a[j]) > kv:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = v


def _sorted_or_reverse(
    a: Sequence[T],
    lo: int,
    hi: int,
    key: Callable[[T], object] | None,
) -> tuple[bool, bool]:
    n = hi - lo
    if n <= 1:
        return True, False

    if key is None:
        def get_key(x):
            return x
    else:
        get_key = key

    first = get_key(a[lo])
    last = get_key(a[hi - 1])

    if first <= last:
        prev = first
        for i in range(lo + 1, hi):
            cur = get_key(a[i])
            if cur < prev:
                break
            prev = cur
        else:
            return True, False

    prev = get_key(a[lo])
    for i in range(lo + 1, hi):
        cur = get_key(a[i])
        if cur > prev:
            break
        prev = cur
    else:
        return False, True

    return False, False



def _choose_splitters(
    a: Sequence[T],
    lo: int,
    hi: int,
    num_buckets: int,
    *,
    key: Callable[[T], object] | None,
) -> list[T]:
    n = hi - lo
    if num_buckets <= 1:
        return []

    if key is None:
        def get_key(x):
            return x
    else:
        get_key = key

    step = _oversampling_factor(n)
    num_samples = step * num_buckets - 1
    num_samples = min(num_samples, n)

    indices = random.sample(range(lo, hi), num_samples)
    sample = [a[i] for i in indices]
    sample.sort(key=get_key)

    if num_samples < num_buckets:
        num_buckets = max(2, num_samples)
    stride = num_samples / num_buckets

    splitters: list[T] = []
    for b in range(1, num_buckets):
        pos = int(b * stride)
        if pos == 0:
            pos = 1
        if pos >= num_samples:
            pos = num_samples - 1
        splitters.append(sample[pos])
    return splitters


def _partition_once(
    a: MutableSequence[T],
    lo: int,
    hi: int,
    *,
    key: Callable[[T], object] | None,
) -> list[tuple[int, int]]:
    n = hi - lo
    if n <= BASE_CASE_SIZE:
        _insertion_sort(a, lo, hi, key)
        return []

    if n <= BASE_CASE_SIZE * (1 << LOG_BUCKETS):
        log_buckets = max(1, _log2_floor(max(2, n // BASE_CASE_SIZE)))
    elif n <= BASE_CASE_SIZE * (1 << (2 * LOG_BUCKETS)):
        log_buckets = max(1, ( _log2_floor(max(2, n // BASE_CASE_SIZE)) + 1 ) // 2)
    else:
        log_buckets = LOG_BUCKETS

    num_buckets = 1 << log_buckets
    num_buckets = min(num_buckets, MAX_BUCKETS)
    if num_buckets < 2:
        _insertion_sort(a, lo, hi, key)
        return []

    if key is None:
        def get_key(x):
            return x
    else:
        get_key = key

    splitters = _choose_splitters(a, lo, hi, num_buckets, key=key)
    splitter_keys = [get_key(s) for s in splitters]

    buckets: list[list[T]] = [[] for _ in range(num_buckets)]

    for i in range(lo, hi):
        v = a[i]
        kv = get_key(v)
        idx = bisect_right(splitter_keys, kv)
        buckets[idx].append(v)

    ranges: list[tuple[int, int]] = []
    idx = lo
    for b in buckets:
        next_idx = idx + len(b)
        a[idx:next_idx] = b
        ranges.append((idx, next_idx))
        idx = next_idx

    return ranges



def _ips4o_sort_recursive(
    a: MutableSequence[T],
    lo: int,
    hi: int,
    *,
    key: Callable[[T], object] | None,
) -> None:
    n = hi - lo
    if n <= BASE_CASE_SIZE:
        _insertion_sort(a, lo, hi, key)
        return

    ranges = _partition_once(a, lo, hi, key=key)
    if not ranges:
        return

    for r_lo, r_hi in ranges:
        if r_hi - r_lo > BASE_CASE_SIZE:
            _ips4o_sort_recursive(a, r_lo, r_hi, key=key)
        else:
            _insertion_sort(a, r_lo, r_hi, key)



def sort(
    a: MutableSequence[T],
    *,
    key: Callable[[T], object] | None = None,
    reverse: bool = False,
) -> None:

    lo, hi = 0, len(a)

    if not reverse:
        is_sorted, is_rev = _sorted_or_reverse(a, lo, hi, key)
        if is_sorted:
            return
        if is_rev:
            a.reverse()
            return

    _ips4o_sort_recursive(a, lo, hi, key=key)
    if reverse:
        a.reverse()



def sort_parallel(
    a: MutableSequence[T],
    *,
    key: Callable[[T], object] | None = None,
    reverse: bool = False,
    max_workers: int | None = 10000,
) -> None:
    n = len(a)
    if n <= PARALLEL_THRESHOLD:
        sort(a, key=key, reverse=reverse)
        return

    lo, hi = 0, n

    if not reverse:
        is_sorted, is_rev = _sorted_or_reverse(a, lo, hi, key)
        if is_sorted:
            return
        if is_rev:
            a.reverse()
            return

    if max_workers is None:
        max_workers = os.cpu_count() or 2

    ranges = _partition_once(a, lo, hi, key=key)
    if not ranges:
        if reverse:
            a.reverse()
        return

    big_ranges: list[tuple[int, int]] = []
    small_ranges: list[tuple[int, int]] = []
    for r_lo, r_hi in ranges:
        if r_hi - r_lo > BASE_CASE_SIZE * 4:
            big_ranges.append((r_lo, r_hi))
        else:
            small_ranges.append((r_lo, r_hi))

    for r_lo, r_hi in small_ranges:
        _ips4o_sort_recursive(a, r_lo, r_hi, key=key)

    if big_ranges:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_ips4o_sort_recursive, a, r_lo, r_hi, key=key)
                for (r_lo, r_hi) in big_ranges
            ]
            for f in futures:
                f.result()

    if reverse:
        a.reverse()
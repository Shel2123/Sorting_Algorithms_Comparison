from random import randint as rd
import time
import sys
from concurrent.futures import ProcessPoolExecutor
sys.setrecursionlimit(10**7)
from src.ips4 import sort_parallel

def powersort(a):
    n = len(a)
    if n <= 1:
        return a

    buf = [0] * n

    def find_run(i, a=a, n=n):
        j = i + 1
        if j >= n:
            return n

        aj_1 = a[j - 1]
        aj = a[j]

        if aj < aj_1:
            while j < n:
                prev = aj
                j += 1
                if j >= n:
                    break
                aj = a[j]
                if aj >= prev:
                    break
            a[i:j] = a[i:j][::-1]
            return j
        else:
            while j < n:
                prev = aj
                j += 1
                if j >= n:
                    break
                aj = a[j]
                if aj < prev:
                    break
            return j

    def merge(l1, r1, l2, r2, a=a, buf=buf):
        i = l1
        j = l2
        k = l1
        ai = a[i]
        aj = a[j]
        b = buf

        while True:
            if aj < ai:
                b[k] = aj
                j += 1
                k += 1
                if j >= r2:
                    break
                aj = a[j]
            else:
                b[k] = ai
                i += 1
                k += 1
                if i >= r1:
                    break
                ai = a[i]

        while i < r1:
            b[k] = a[i]
            i += 1
            k += 1

        while j < r2:
            b[k] = a[j]
            j += 1
            k += 1

        a[l1:r2] = b[l1:r2]
        return l1, r2

    SCALE_BITS = 32

    def node_power(b1, e1, b2, e2, n=n, SCALE_BITS=SCALE_BITS):
        mid1 = b1 + e1
        mid2 = b2 + e2

        v1 = (mid1 << SCALE_BITS) // (2 * n)
        v2 = (mid2 << SCALE_BITS) // (2 * n)

        x = v1 ^ v2
        if x == 0:
            return SCALE_BITS
        return SCALE_BITS - x.bit_length()

    stack = []
    stack_append = stack.append
    stack_pop = stack.pop

    b1 = 0
    e1 = find_run(0)

    while e1 < n:
        b2 = e1
        e2 = find_run(b2)

        P = node_power(b1, e1, b2, e2)

        while stack and stack[-1][2] > P:
            l, r, _ = stack_pop()
            b1, e1 = merge(l, r, b1, e1)

        stack_append((b1, e1, P))
        b1, e1 = b2, e2

    while stack:
        l, r, _ = stack_pop()
        b1, e1 = merge(l, r, b1, e1)

    return a


def radix_sort(a):
    n = len(a)
    if n <= 1:
        return a


    b = [0] * n
    for shift in (0, 8, 16, 24):
        cnt = [0] * 256

        for x in a:
            cnt[(x >> shift) & 255] += 1

        s = 0
        for i in range(256):
            c = cnt[i]
            cnt[i] = s
            s += c

        for x in a:
            idx = (x >> shift) & 255
            b[cnt[idx]] = x
            cnt[idx] += 1

        a, b = b, a


    return a


def insertion_sort(arr):
    n = len(arr)
    for i in range(1, n):
        x = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > x:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = x
    return arr


def bucketish_sort(lst, factor=4):
    n = len(lst)
    if n == 0:
        return []
    if n == 1:
        return lst[:]

    mn = min(lst)
    mx = max(lst)
    span = mx - mn

    if span == 0:
        return lst[:]

    size = factor * n
    size_minus_1 = size - 1

    buckets = {}

    inv_span = 1.0 / span
    for x in lst:
        norm = (x - mn) * inv_span
        idx = int(norm * size_minus_1)

        if idx < 0:
            idx = 0
        elif idx >= size:
            idx = size_minus_1

        if idx in buckets:
            buckets[idx].append(x)
        else:
            buckets[idx] = [x]

    out = []
    for idx in sorted(buckets.keys()):
        b = buckets[idx]
        if len(b) > 1:
            b = insertion_sort(b)
        out.extend(b)

    return out


def merge_sort(arr):
    n = len(arr)
    if n <= 1:
        return arr

    tmp = [None] * n
    def _merge_sort(l, r):
        if r - l <= 1:
            return

        m = (l + r) // 2
        _merge_sort(l, m)
        _merge_sort(m, r)

        i, j, k = l, m, l
        while i < m and j < r:
            if arr[i] <= arr[j]:
                tmp[k] = arr[i]
                i += 1
            else:
                tmp[k] = arr[j]
                j += 1
            k += 1

        while i < m:
            tmp[k] = arr[i]
            i += 1
            k += 1

        while j < r:
            tmp[k] = arr[j]
            j += 1
            k += 1

        for i in range(l, r):
            arr[i] = tmp[i]

    _merge_sort(0, n)
    return arr

def default_sort(arr):
    arr.sort()
    return arr

def measure(sort_fn, base_arr, reps=3):
    best = float('inf')
    for _ in range(reps):
        arr = base_arr.copy()
        start = time.perf_counter()
        sort_fn(arr)
        end = time.perf_counter()
        best = min(best, end - start)
    return best

def bench_one_n(args):
    n, base_arr = args

    t_radix = measure(radix_sort, base_arr, reps=3)
    t_bucket = measure(bucketish_sort, base_arr, reps=3)
    t_power = measure(powersort, base_arr, reps=3)
    t_merge = measure(merge_sort, base_arr, reps=3)
    t_ips4 = measure(sort_parallel, base_arr, reps=3)
    t_default = measure(default_sort, base_arr, reps=3)

    return n, t_radix, t_bucket, t_power, t_merge, t_ips4, t_default

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sizes = list(range(0, 100_000, 10_000))

    tasks = []
    for n in sizes:
        base_arr = [rd(1, 10_000_000) for _ in range(n)]
        tasks.append((n, base_arr))

    times_radix = []
    times_bucket = []
    times_power = []
    times_merge = []
    times_ips = []
    times_default = []

    with ProcessPoolExecutor() as executor:
        for n, t_radix, t_bucket, t_power, t_merge, t_ips, t_default in executor.map(bench_one_n, tasks):
            times_radix.append(t_radix)
            times_bucket.append(t_bucket)
            times_power.append(t_power)
            times_merge.append(t_merge)
            times_ips.append(t_ips)
            times_default.append(t_default)

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_radix,  label="radix_sort")
    plt.plot(sizes, times_bucket, label="bucketish_sort")
    plt.plot(sizes, times_power,  label="power_sort")
    plt.plot(sizes, times_merge,  label="merge_sort")
    plt.plot(sizes, times_ips, label="ips4o")
    plt.plot(sizes, times_default, label=".sort()")
    plt.xlabel("Array size")
    plt.ylabel("Time in sec")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
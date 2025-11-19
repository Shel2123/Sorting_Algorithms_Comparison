from random import randint as rd
import random
import time
import sys
from concurrent.futures import ProcessPoolExecutor
from src.ips4 import sort_parallel
import matplotlib.pyplot as plt

sys.setrecursionlimit(10**7)

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
    if len(base_arr) <= 1:
        return 0.0
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

def run_bench(tasks):
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

    return times_radix, times_bucket, times_power, times_merge, times_ips, times_default

def trend_with_jumps(n, jump_prob=0.05):
    arr = []
    value = 1
    for _ in range(n):
        if random.random() < jump_prob:
            value += rd(-10, 10)
        else:
            value += rd(0, 1)

        if value < 1:
            value = 1

        arr.append(value)

    return arr

def worst_case(n):
    return list(range(n, 0, -1))

def best_case(n):
    return list(range(n))

def worst_case_alternating_high_low(n):
    high = list(range(n, 0, -1))
    low = list(range(1, n + 1))
    arr = []
    for h, l in zip(high, low):
        arr.append(h)
        arr.append(l)
    return arr[:n]

def generate_many_duplicates(n, distinct_values=3, max_value=20):
    base_values = random.sample(range(1, max_value + 1), k=distinct_values)
    return [random.choice(base_values) for _ in range(n)]


def generate_many_unique_spread(n, range_multiplier=1000):
    """
    range_multiplier - "range" of values will be n * range_multiplier
    """
    max_value = n * range_multiplier
    arr = random.sample(range(1, max_value + 1), n)
    return arr


def plot_results(sizes, series, title):
    """
    sizes - list of array sizes
    series - list of tuples (label, values), where values is a list of times corresponding to sizes
    title  - title of the plot
    """
    plt.figure(figsize=(10, 6))
    for label, values in series:
        plt.plot(sizes, values, label=label)

    plt.title(title)
    plt.xlabel("Array size")
    plt.ylabel("Time, sec")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    sizes = list(range(1, 1_000_000, 100_000))

    tasks_random = []
    for n in sizes:
        base_arr = [rd(1, 10_000_000) for _ in range(n)]
        tasks_random.append((n, base_arr))

    (times_radix_rand,
     times_bucket_rand,
     times_power_rand,
     times_merge_rand,
     times_ips_rand,
     times_default_rand) = run_bench(tasks_random)

    plot_results(
        sizes,
        [
            ("radix_sort",   times_radix_rand),
            ("bucketish_sort", times_bucket_rand),
            ("power_sort",   times_power_rand),
            ("merge_sort",   times_merge_rand),
            ("ips4o",        times_ips_rand),
            (".sort()",      times_default_rand),
        ],
        "Random data sorting comparison",
    )

    tasks_jumps = []
    for n in sizes:
        base_arr = trend_with_jumps(n, jump_prob=0.05)
        tasks_jumps.append((n, base_arr))

    (times_radix_jump,
     times_bucket_jump,
     times_power_jump,
     times_merge_jump,
     times_ips_jump,
     times_default_jump) = run_bench(tasks_jumps)

    plot_results(
        sizes,
        [
            ("radix_sort",   times_radix_jump),
            ("bucketish_sort", times_bucket_jump),
            ("power_sort",   times_power_jump),
            ("merge_sort",   times_merge_jump),
            ("ips4o",        times_ips_jump),
            (".sort()",      times_default_jump),
        ],
        "Data with jumps sorting comparison",
    )

    tasks_best = []
    for n in sizes:
        base_arr = best_case(n)
        tasks_best.append((n, base_arr))

    (times_radix_best,
     times_bucket_best,
     times_power_best,
     times_merge_best,
     times_ips_best,
     times_default_best) = run_bench(tasks_best)

    plot_results(
        sizes,
        [
            ("radix_sort", times_radix_best),
            ("bucketish_sort", times_bucket_best),
            ("power_sort", times_power_best),
            ("merge_sort", times_merge_best),
            ("ips4o", times_ips_best),
            (".sort()", times_default_best),
        ],
        "Best-case data sorting comparison",
    )

    tasks_worst = []
    for n in sizes:
        base_arr = worst_case(n)
        tasks_worst.append((n, base_arr))

    (times_radix_worst,
     times_bucket_worst,
     times_power_worst,
     times_merge_worst,
     times_ips_worst,
     times_default_worst) = run_bench(tasks_worst)

    plot_results(
        sizes,
        [
            ("radix_sort",   times_radix_worst),
            ("bucketish_sort", times_bucket_worst),
            ("power_sort",   times_power_worst),
            ("merge_sort",   times_merge_worst),
            ("ips4o",        times_ips_worst),
            (".sort()",      times_default_worst),
        ],
        "Worst-case data sorting comparison",
    )

    tasks_alternating = []
    for n in sizes:
        base_arr = worst_case_alternating_high_low(n)
        tasks_alternating.append((n, base_arr))

    (times_radix_alternating,
     times_bucket_alternating,
     times_power_alternating,
     times_merge_alternating,
     times_ips_alternating,
     times_default_alternating) = run_bench(tasks_alternating)

    plot_results(
        sizes,
        [
            ("radix_sort", times_radix_alternating),
            ("bucketish_sort", times_bucket_alternating),
            ("power_sort", times_power_alternating),
            ("merge_sort", times_merge_alternating),
            ("ips4o", times_ips_alternating),
            (".sort()", times_default_alternating),
        ],
        "Alternating-case data sorting comparison",
    )
    # task_duplicates = []
    # for n in sizes:
    #     base_arr = generate_many_duplicates(n, distinct_values=3, max_value=20)
    #     print(base_arr)
    #     task_duplicates.append((n, base_arr))
    #
    # print("Starting benchmark for many duplicates data...")
    #
    # (times_radix_duplicates,
    #  times_bucket_duplicates,
    #  times_power_duplicates,
    #  times_merge_duplicates,
    #  times_ips_duplicates,
    #  times_default_duplicates) = run_bench(task_duplicates)
    #
    # print("Benchmark for many duplicates data completed.")
    #
    # plot_results(
    #     sizes,
    #     [
    #         ("radix_sort", times_radix_duplicates),
    #         ("bucketish_sort", times_bucket_duplicates),
    #         ("power_sort", times_power_duplicates),
    #         ("merge_sort", times_merge_duplicates),
    #         ("ips4o", times_ips_duplicates),
    #         (".sort()", times_default_duplicates),
    #     ],
    #     "Many Duplicates Data Sorting Comparison",
    # )


    task_unique_spread = []
    for n in sizes:
        base_arr = generate_many_unique_spread(n, range_multiplier=1000)
        task_unique_spread.append((n, base_arr))

    (times_radix_unique,
     times_bucket_unique,
     times_power_unique,
     times_merge_unique,
     times_ips_unique,
     times_default_unique) = run_bench(task_unique_spread)

    plot_results(
        sizes,
        [
            ("radix_sort", times_radix_unique),
            ("bucketish_sort", times_bucket_unique),
            ("power_sort", times_power_unique),
            ("merge_sort", times_merge_unique),
            ("ips4o", times_ips_unique),
            (".sort()", times_default_unique),
        ],
        "Many Unique Spread Data Sorting Comparison",
    )

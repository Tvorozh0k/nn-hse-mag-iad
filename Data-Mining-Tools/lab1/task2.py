import time
import heapq
import random
import numpy as np

from tqdm import tqdm
from collections import defaultdict


def get_median(max_heap, min_heap, max_heap_sz, min_heap_sz):
    if max_heap_sz == min_heap_sz:
        return (-max_heap[0] + min_heap[0]) / 2
    else:
        return -max_heap[0]


def median_stream(arr: np.ndarray, k: int):
    """
    Efficient implementation of algorithm, which
    finds all medians across the sliding window of size k
    :param arr: numpy array
    :param k: the size of sliding window
    :return: the array of all found medians
    """

    res = np.empty(arr.shape[0] - k + 1, dtype=np.float32)

    lst = arr.tolist()

    max_heap, max_heap_sz = [], 0
    min_heap, min_heap_sz = [], 0

    for i in range(k):
        # insertion
        if max_heap_sz == 0 or lst[i] <= get_median(max_heap, min_heap, max_heap_sz, min_heap_sz):
            heapq.heappush(max_heap, -lst[i])
            max_heap_sz += 1
        else:
            heapq.heappush(min_heap, lst[i])
            min_heap_sz += 1

        # balancing
        if max_heap_sz - min_heap_sz == 2:
            # rotation right
            root = -heapq.heappop(max_heap)
            max_heap_sz -= 1
            heapq.heappush(min_heap, root)
            min_heap_sz += 1
        elif min_heap_sz - max_heap_sz == 1:
            # rotation left
            root = heapq.heappop(min_heap)
            min_heap_sz -= 1
            heapq.heappush(max_heap, -root)
            max_heap_sz += 1

    removal_debt = defaultdict(int)

    res[0] = get_median(max_heap, min_heap, max_heap_sz, min_heap_sz)
    for i in range(k, arr.shape[0]):
        removal_debt[lst[i - k]] += 1

        if lst[i - k] <= get_median(max_heap, min_heap, max_heap_sz, min_heap_sz):
            max_heap_sz -= 1
        else:
            min_heap_sz -= 1

        # balancing
        if max_heap_sz - min_heap_sz == 2:
            # rotation right
            root = -heapq.heappop(max_heap)
            max_heap_sz -= 1
            heapq.heappush(min_heap, root)
            min_heap_sz += 1
        elif min_heap_sz - max_heap_sz == 1:
            # rotation left
            root = heapq.heappop(min_heap)
            min_heap_sz -= 1
            heapq.heappush(max_heap, -root)
            max_heap_sz += 1

        lazy_removal_flag = True
        while lazy_removal_flag:
            if len(max_heap) > 0 and removal_debt[-max_heap[0]] > 0:
                removal_debt[-max_heap[0]] -= 1
                heapq.heappop(max_heap)
            elif len(min_heap) > 0 and removal_debt[min_heap[0]] > 0:
                removal_debt[min_heap[0]] -= 1
                heapq.heappop(min_heap)
            else:
                lazy_removal_flag = False

        # insertion
        if max_heap_sz == 0 or lst[i] <= get_median(max_heap, min_heap, max_heap_sz, min_heap_sz):
            heapq.heappush(max_heap, -lst[i])
            max_heap_sz += 1
        else:
            heapq.heappush(min_heap, lst[i])
            min_heap_sz += 1

        # balancing
        if max_heap_sz - min_heap_sz == 2:
            # rotation right
            root = -heapq.heappop(max_heap)
            max_heap_sz -= 1
            heapq.heappush(min_heap, root)
            min_heap_sz += 1
        elif min_heap_sz - max_heap_sz == 1:
            # rotation left
            root = heapq.heappop(min_heap)
            min_heap_sz -= 1
            heapq.heappush(max_heap, -root)
            max_heap_sz += 1

        lazy_removal_flag = True
        while lazy_removal_flag:
            if len(max_heap) > 0 and removal_debt[-max_heap[0]] > 0:
                removal_debt[-max_heap[0]] -= 1
                heapq.heappop(max_heap)
            elif len(min_heap) > 0 and removal_debt[min_heap[0]] > 0:
                removal_debt[min_heap[0]] -= 1
                heapq.heappop(min_heap)
            else:
                lazy_removal_flag = False

        res[i - k + 1] = get_median(max_heap, min_heap, max_heap_sz, min_heap_sz)

    return res


def median_stream_naive(arr: np.ndarray, k: int):
    """
    Naive implementation of algorithm, essential to
    test the correctness the efficient implementation of
    algorithm
    :param arr: numpy array
    :param k: the size of sliding window
    :return: the array of all found medians
    """

    res = np.empty(arr.shape[0] - k + 1, dtype=np.float32)

    for i in range(arr.shape[0] - k + 1):
        res[i] = np.median(arr[i:i+k]).item()

    return res


def gen_input(n: int):
    arr = np.random.randint(low=-10000, high=10000, size=n)
    k = random.randint(2, n)

    return arr, k


def perf_measurement():
    # Test correctness:
    # for _ in tqdm(range(100), desc='eval'):
    #     arr, k = gen_input(10000)
    #     expected = median_stream_naive(arr, k)
    #     actual = median_stream(arr, k)
    #
    #     if not np.array_equal(expected, actual):
    #         print(arr, k)
    #         print(expected)
    #         print(actual)
    #         break

    arr, k = gen_input(10000)

    # warmup
    for _ in range(10):
        median_stream(arr, k)

    n_iter = 100
    t0 = time.perf_counter()
    for _ in range(n_iter):
        median_stream(arr, k)

    print(f"Perf time: {(time.perf_counter() - t0) / n_iter} seconds")


def main():
    # Simple test:
    # nums = np.array([1, 3, -1, -3, 5, 3, 6, 7])
    # k = 3
    #
    # expected = median_stream_naive(nums, k)
    # actual = median_stream(nums, k)
    #
    # print(f'Expected: {expected}')
    # print(f'Actual: {actual}')

    perf_measurement()


if __name__ == '__main__':
    main()

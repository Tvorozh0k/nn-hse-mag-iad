import time
import numpy as np
import numba
import numpy.random

from tqdm import tqdm


@numba.jit
def match_timestamps(timestamps1: np.ndarray, timestamps2: np.ndarray) -> np.ndarray:
    """
    Timestamp matching function. It returns such array `matching` of length len(timestamps1),
    that for each index i of timestamps1 the element matching[i] contains
    the index j of timestamps2, so that the difference between
    timestamps2[j] and timestamps1[i] is minimal.
    Example:
        timestamps1 = [0, 0.091, 0.5]
        timestamps2 = [0.001, 0.09, 0.12, 0.6]
        => matching = [0, 1, 3]
    """

    matching = np.empty(timestamps1.shape[0], dtype=np.uint32)

    ind = 0
    for i in range(timestamps1.shape[0]):
        while ind < timestamps2.shape[0] and timestamps2[ind] < timestamps1[i]:
            ind += 1

        # ind = len(timestamps2) or timestamp2[ind] >= timestamp1[i]
        if ind == 0:
            matching[i] = ind
        elif ind == timestamps2.shape[0]:
            matching[i] = ind - 1
        else:
            #  timestamps1[i] \in ( timestamps2[ind - 1], timestamps2[ind] ]
            matching[i] = ind if timestamps2[ind] - timestamps1[i] < timestamps1[i] - timestamps2[ind - 1] else ind - 1

    return matching


def match_timestamps_correct(timestamps1: np.ndarray, timestamps2: np.ndarray) -> np.ndarray:
    matching = np.empty(timestamps1.shape[0], dtype=np.uint32)

    for i in range(timestamps1.shape[0]):
        cur = np.searchsorted(timestamps2, timestamps1[i])
        ind, dist = -1, 1e9 + 7

        if cur > 0 and abs(timestamps1[i] - timestamps2[cur - 1]) < dist:
            ind = cur - 1
            dist = abs(timestamps1[i] - timestamps2[cur - 1])

        if cur < timestamps2.shape[0] and abs(timestamps1[i] - timestamps2[cur]) < dist:
            ind = cur
            dist = abs(timestamps1[i] - timestamps2[cur])

        if cur + 1 < timestamps2.shape[0] and abs(timestamps1[i] - timestamps2[cur + 1]) < dist:
            ind = cur + 1
            dist = abs(timestamps1[i] - timestamps2[cur + 1])

        matching[i] = ind

    return matching


def make_timestamps(fps: int, st_ts: float, fn_ts: float) -> np.ndarray:
    """
    Create array of timestamps. This array is discretized with fps,
    but not evenly.
    Timestamps are assumed sorted nad unique.
    Parameters:
    - fps: int
        Average frame per second
    - st_ts: float
        First timestamp in the sequence
    - fn_ts: float
        Last timestamp in the sequence
    Returns:
        np.ndarray: synthetic timestamps
    """
    # generate uniform timestamps
    timestamps = np.linspace(st_ts, fn_ts, int((fn_ts - st_ts) * fps))
    # add a fps noise
    timestamps += np.random.randn(len(timestamps))
    timestamps = np.unique(np.sort(timestamps))
    return timestamps


def check_correctness():
    st_ts = time.time()
    fn_ts = st_ts + 3600 * 2
    fps = 30

    for i in tqdm(range(100)):
        ts1 = make_timestamps(fps, st_ts, fn_ts)
        ts2 = make_timestamps(fps, st_ts + 200, fn_ts)

        expected = match_timestamps_correct(ts1, ts2)
        actual = match_timestamps(ts1, ts2)

        if not np.array_equal(expected, actual):
            print(ts1)
            print(ts2)
            print(expected)
            print(actual)
            break


def perf_measurement():
    """
    Performance measurement procedure
    """
    st_ts = time.time()
    fn_ts = st_ts + 3600 * 2
    fps = 30
    ts1 = make_timestamps(fps, st_ts, fn_ts)
    ts2 = make_timestamps(fps, st_ts + 200, fn_ts)
    # warmup
    for _ in range(10):
        match_timestamps(ts1, ts2)
    n_iter = 100
    t0 = time.perf_counter()
    for _ in range(n_iter):
        match_timestamps(ts1, ts2)
    print(f"Perf time: {(time.perf_counter() - t0) / n_iter} seconds")


def main():
    """
    Setup:
        Say we have two videocameras, each filming the same scene. We make
        a prediction based on this scene (e.g. detect a human pose).
        To improve the robustness of the detection algorithm,
        we average the predictions from both cameras at each moment.
        The camera data is a pair (frame, timestamp), where the timestamp
        represents the moment when the frame was captured by the camera.

    Problem:
        For each frame of camera1, we need to find the index of the
        corresponding frame received by camera2. The frame i from camera2
        corresponds to the frame j from camera1, if
        abs(timestamps[i] - timestamps[j]) is minimal for all i.

    Estimation criteria:
        - The solution has to be optimal algorithmically. As an example, let's assume that
    the best solution has O(n^3) complexity. In this case, the O(n^3 * logn) solution will add -1 point penalty,
    O(n^4) will add -2 points and so on.
        - The solution has to be optimal python-wise.
    If it can be optimized ~x5 times by rewriting the algorithm in Python,
    this will add -1 point. x20 times optimization will result in -2 points.
    You may use any optimization library!
        - All corner cases must be handled correctly. A wrong solution will add -3 points.
        - The base score is 6.
        - Parallel implementation adds +1 point, provided it is effective (cannot be optimized x5 times)
        - 3 points for this homework are added by completing the second problem (the one with the medians).
    Optimize the solution to work with ~2-3 hours of data.
    Good luck!
    """

    perf_measurement()
    # check_correctness()


if __name__ == '__main__':
    main()

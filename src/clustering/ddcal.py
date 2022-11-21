import numpy as np
import math


def _normalize_min_max_feature(data):
    """
    helper
    normalizer by using feature scaling
    O(n)
    https://en.wikipedia.org/wiki/Feature_scaling
    """
    min_value = np.amin(data)
    max_value = np.amax(data)
    if min_value != max_value:
        return np.array([(x - min_value) / (max_value - min_value) for x in data])
    else:
        return np.array([float(0) for x in data])


class DDCAL:
    def __init__(
            self,
            n_clusters: int = 10,
            feature_boundary_min: float = 0.1,
            feature_boundary_max: float = 0.45,
            num_simulations: int = 20,
            q_tolerance: float = 0.45,
            q_tolerance_increase_step: float = 0.5,
    ):
        """
        :param n_clusters:
        number of aimed clusters
        e.g., 10 means, that 10 clusters are aimed to be built
        :param
        feature_boundary_min:
        starting simulation boundary
        e.g., 0.1 means that 10% outliers are considered as potential clusters in upper and lower
        bound
        0.45 means that 45% outliers are considered as potential clusters in upper and lower
        bound
        :param feature_boundary_max:
        end simulation boundary
        :param num_simulations:
        number of simulations, where starting and end simulation boundary are
        sliced into evenly spaced boundaries and tested for clustering
        e.g., 20 means that 20 boundaries are tested to meet an even distribution of clusters
        :param q_tolerance:
        tolerance factor of element in cluster for even distribution of clusters,
        e.g., 0.45 means 45% tolerance factor
        :param q_tolerance_increase_step: increase step of tolerance factor in iteration if for
        simulations of boundaries from starting to end,the tolerance factor was not met
        e.g., 0.5 means, that the set tolerance_factor will be increased by 50%
        """

        self.labels_sorted_data: np.ndarray = None
        self.sorted_data: np.ndarray = None
        self.n_clusters = n_clusters
        self.feature_boundary_min = feature_boundary_min
        self.feature_boundary_max = feature_boundary_max
        self.num_simulations = num_simulations
        self.q_tolerance = q_tolerance
        self.q_tolerance_increase_step = q_tolerance_increase_step

    def fit(self, data: np.ndarray):
        """
        ddacl clustering 1d function
        O( n log n)

        :param data: 1d numpy array
        :return:
        setting parameters for clustered results of 1d numpy array
        self.labels_sorted_data
        self.sorted_data
        """

        q_tolerance_tmp = self.q_tolerance
        clusters_array = np.arange(0, self.n_clusters)
        # prepare data
        data = np.sort(data, kind='mergesort')

        data_size = data.size
        data_min_idx = 0
        data_max_idx = data.size

        labels_ = np.zeros(data_size)  # result
        labels_.fill(-1)

        q_aim = data_size / self.n_clusters

        while clusters_array.size > 0 and data[data_min_idx:data_max_idx].size > 0:

            norm_frequencies = _normalize_min_max_feature(data[data_min_idx:data_max_idx])

            # if q_aim_up enabled, SED decreasing (negative effect) but SCSD (sum variance) decreasing (positive effect)
            # q_aim_up = (q_aim + q_aim * q_tolerance_tmp)  # test: disabled (because we start with low percentage)
            q_aim_lo = (q_aim - q_aim * q_tolerance_tmp)

            q_idx_best_found_lo = False
            q_idx_best_found_up = False

            if self.feature_boundary_min == self.feature_boundary_max:
                self.num_simulations = 1
            boundaries = np.linspace(self.feature_boundary_min, self.feature_boundary_max, num=self.num_simulations)
            for b in range(0, boundaries.size):
                # debug_curr_boundary = boundaries[b]
                q_lower_bound = 0
                q_upper_bound = 0
                it = np.nditer(norm_frequencies, flags=['f_index'])
                while not it.finished:  # O(log n)
                    # get x% lower bound of data
                    if norm_frequencies[it.index] <= boundaries[b]:
                        q_lower_bound += 1

                    # get x% upper bound of data
                    if norm_frequencies[it.index] >= (1 - boundaries[b]):
                        q_upper_bound += 1

                    it.iternext()

                # print("q_upper_bound:{0} and q_lower_bound:{1}".format(q_upper_bound, q_lower_bound))
                q_lower_bound_diff = math.nan
                q_upper_bound_diff = math.nan

                def get_next_gap_q_lower_bound(_data: np.ndarray, current_idx: int, min_idx: int, max_idx: int) \
                        -> tuple[float, int]:
                    """
                    helper
                    :return: next gap value outside boundary, new index
                    """
                    pointer = current_idx
                    _next_gap = 0
                    while True:
                        pointer += 1
                        if pointer <= min_idx or pointer >= max_idx:
                            return _next_gap, pointer

                        _next_gap = abs(data[pointer] - data[current_idx])

                        if _next_gap > 0:  # filter duplicate values
                            return _next_gap, pointer

                def get_next_gap_q_upper_bound(_data: np.ndarray, current_idx: int, min_idx: int, max_idx: int) \
                        -> tuple[float, int]:
                    """
                    helper
                    :return: next gap value outside boundary, new index
                    """
                    pointer = current_idx
                    _next_gap = 0
                    while True:
                        pointer -= 1
                        if pointer <= min_idx or pointer >= max_idx:
                            return _next_gap, pointer

                        _next_gap = abs(data[pointer] - data[current_idx])

                        if _next_gap > 0:  # filter duplicate values
                            return _next_gap, pointer

                # optimize: add nearby elements by using standard deviation
                while True:
                    if data[data_min_idx:data_min_idx + q_lower_bound].size == 0:  # skip if array is empty
                        break
                    std_dev_current = np.std(data[data_min_idx:data_min_idx + q_lower_bound])
                    next_gap, idx_next_gap = get_next_gap_q_lower_bound(data, data_min_idx + q_lower_bound,
                                                                        data_min_idx, data_max_idx - 1)
                    std_dev_next = np.std(data[data_min_idx:idx_next_gap])
                    if std_dev_next < std_dev_current:
                        q_lower_bound += idx_next_gap - (data_min_idx + q_lower_bound)
                    else:
                        break

                while True:  # O(log n)
                    if data[data_max_idx - q_upper_bound:data_max_idx].size == 0:  # skip if array is empty
                        break
                    std_dev_current = np.std(data[data_max_idx - q_upper_bound:data_max_idx])
                    next_gap, idx_next_gap = get_next_gap_q_upper_bound(data, data_max_idx - q_upper_bound,
                                                                        data_min_idx, data_max_idx - 1)
                    std_dev_next = np.std(data[idx_next_gap:data_max_idx])
                    if std_dev_next < std_dev_current:
                        q_upper_bound += data_max_idx - (idx_next_gap + q_upper_bound)
                    else:
                        break

                if q_aim_lo <= q_lower_bound:  # <= q_aim_up:
                    q_lower_bound_diff = abs(q_lower_bound - q_aim)
                if q_aim_lo <= q_upper_bound:  # <= q_aim_up:
                    q_upper_bound_diff = abs(q_upper_bound - q_aim)

                if math.isnan(q_lower_bound_diff) or math.isnan(q_upper_bound_diff):
                    # print("debug: no cluster found for boundary: {0} and tolerance: {1} testing next boundary".format(boundaries[b], q_tolerance_tmp))
                    continue  # to start over slicing the boundary range and testing with extended q tolerance

                # favour min q - add the smallest list in final results list and delete it from input list
                else:
                    # test: disabled (because we start with low percentage)
                    # if math.isnan(q_lower_bound_diff):
                    #     q_idx_best_found_up = True
                    #     break
                    # elif math.isnan(q_upper_bound_diff):
                    #     q_idx_best_found_lo = True
                    #     break
                    # elif q_lower_bound_diff <= q_upper_bound_diff:
                    if q_lower_bound_diff <= q_upper_bound_diff:
                        q_idx_best_found_lo = True
                        break
                    else:
                        q_idx_best_found_up = True
                        break

            # CONTINUE HERE AFTER break OR all boundaries were tested (q_idx_best_found_lo/up always false)
            if not q_idx_best_found_lo and not q_idx_best_found_up:  # both boundaries must be greater than min q_aim
                # print("no boundary found for all tested boundaries: extending q tolerance")
                q_tolerance_tmp = q_tolerance_tmp + q_tolerance_tmp * self.q_tolerance_increase_step
                continue  # to start over slicing the boundary range and testing with extended q tolerance

            if q_idx_best_found_lo:
                # print("debug: adding {0} elements from lower bound to results".format(q_lower_bound))
                labels_[data_min_idx:data_min_idx + q_lower_bound] = clusters_array[0]
                data_min_idx = data_min_idx + q_lower_bound  # update data index

                clusters_array = np.delete(clusters_array, 0)

            # add elements to upper bucket
            else:
                # print("debug: adding {0} elements from upper bound to results".format(q_upper_bound))
                labels_[data_max_idx - q_upper_bound:data_max_idx] = clusters_array[-1]
                data_max_idx = data_max_idx - q_upper_bound  # update data index

                clusters_array = np.delete(clusters_array, clusters_array.size - 1)

            # last run put all remaining values to last bucket list element
            if clusters_array.size == 1:
                labels_[data_min_idx:data_max_idx] = clusters_array[0]
                break

            q_tolerance_tmp = self.q_tolerance  # reset q_tolerance
            q_aim = data[data_min_idx:data_max_idx].size / clusters_array.size
            # print("debug: new q aim is {0}".format(q_aim))

        # print("Debug: number of elements in result={0}".format(data_result.size))
        self.sorted_data = data
        self.labels_sorted_data = labels_

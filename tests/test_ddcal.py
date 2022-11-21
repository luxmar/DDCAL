import numpy as np
from src.clustering.ddcal import DDCAL
from tests.analyze import analyze_results
import unittest


class TestDDCAL(unittest.TestCase):
    nuc = None
    sed = None
    sv = None
    msc = None

    ###########
    # init data
    ###########
    cluster_size = 10  # max clusters

    print("max given clusters: {0}".format(cluster_size))
    # read a test data set
    frequencies = np.load('./data/dist_normal_v2.npy')  # q_tolerance=0.45
    # frequencies = np.load('./data/dist_gumbel_v2.npy')  # q_tolerance=0.45
    # frequencies = np.load('./data/dist_uniform_v2.npy')  # q_tolerance=0.1 because uniform
    # frequencies = np.load('./data/dist_exponential_v2.npy')  # q_tolerance=0.1 because one sided dist)
    # frequencies = np.load('./data/dist_2_normal_peaks_v2.npy')  # q_tolerance=0.45
    print("total number of loaded elements={0}\n".format(frequencies.size))

    ###########
    # execution
    ###########
    # wide ranging feature boundary starting with minimum value reduces variance SV and
    # high q tolerance increases distribution SED (and increases chance to use all max clusters) and may decreases SV
    # default: feature_boundary_min=0.1, feature_bouq_tolerance=0.45ndary_max=0.49, num_simulations=20, q_tolerance=0.1, q_tolerance_increase_step=0.5
    ddcal = DDCAL(n_clusters=cluster_size, feature_boundary_min=0.1, feature_boundary_max=0.49,
                  num_simulations=20, q_tolerance=0.45, q_tolerance_increase_step=0.5)
    ddcal.fit(frequencies)

    def test_parameters(self):
        # create frequencies result by matching the labels of clusters to the frequencies
        dt = np.dtype([('bucket', int), ('freq', np.float64)])
        frequencies_result = np.zeros(self.ddcal.sorted_data.size, dtype=dt)
        frequencies_result_count = -1
        for idx_f, f in enumerate(self.ddcal.sorted_data):
            tmp = np.array([(self.ddcal.labels_sorted_data[idx_f], f)], dtype=dt)
            frequencies_result_count = frequencies_result_count + 1
            frequencies_result[frequencies_result_count] = tmp
        print("clusters with resulting frequencies: \n{0}".format(frequencies_result))

        self.assertIsNone(self.nuc)
        self.assertIsNone(self.sed)
        self.assertIsNone(self.sv)
        self.assertIsNone(self.msc)

        self.nuc, self.sed, self.sv, self.msc = analyze_results(frequencies_result, self.cluster_size)

        self.assertIsNotNone(self.nuc)
        self.assertIsNotNone(self.sed)
        self.assertIsNotNone(self.sv)
        self.assertIsNotNone(self.msc)
        assert self.nuc > 0
        assert self.sed > 0
        assert self.sv > 0
        assert self.msc >= -1
        assert self.msc <= 1

    def test_sizes(self):
        self.assertEqual(self.frequencies.size, self.ddcal.sorted_data.size)
        self.assertEqual(self.ddcal.sorted_data.size, self.ddcal.labels_sorted_data.size)

    def test_all_classified(self):
        # labels are initialized with -1 in DDCAL but must not contain -1 after execution of algorithm
        if -1 in self.ddcal.labels_sorted_data:
            assert False
        else:
            assert True

    def test_sorted_data(self):
        is_sorted = lambda a: np.all(a[:-1] <= a[1:])

        if is_sorted(self.ddcal.sorted_data):
            assert True


if __name__ == '__main__':
    unittest.main()

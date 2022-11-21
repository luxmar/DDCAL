import numpy
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

import statistics
import math


# variance: https://stackoverflow.com/questions/26269512/python-numpy-var-returning-wrong-values
# we use the population variance instead of the sample variance because we have all data!
# ddof=1

def analyze_results(results: np.dtype([('bucket', int), ('freq', np.float64)]), size_aimed_clusters: int) \
        -> (float, np.float64, np.float64, np.float64):
    """
    calculating different scores for algorithm (used buckets, distribution to buckets

    :param results: results in particular format
    :param size_aimed_clusters: number of aimed clusters
    :return:
    nuc number of used clusters
    sed score even distribution of clusters
    sv standard deviation of clusters
    silhouette_avg mean silhouette coefficient
    """

    results = np.sort(results, order=['bucket', 'freq'])

    print("\n\nanalyzing the results:\nnumber of max cluster size: {0}".format(size_aimed_clusters))
    size = np.size(np.unique(results['bucket']))
    print("number of actually used clusters: {0}".format(size))
    print("min value of all elements: {0}".format(np.min(results['freq'])))
    print("max value of all elements: {0}".format(np.max(results['freq'])))

    print("mean of all elements: {0}".format(statistics.mean(results['freq'])))
    print("median of all elements: {0}".format(statistics.median(results['freq'])))
    # print("mode of all elements: {0}".format(statistics.mode(results['freq'])))
    print("standard deviation of all elements: {0}".format(math.sqrt(np.var(results['freq']))))
    print("variance of all elements: {0}".format(np.var(results['freq'], ddof=1)))

    print("distribution of all clusters: {0}\n".format(results['bucket']))
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html
    unique_used_buckets, counts_used_buckets = np.unique(results['bucket'], return_counts=True)
    sv = 0  # sum buckets variance
    hom_sum = 0
    for x in unique_used_buckets:
        # check if std dev can be calculated
        mean_bucket = np.around(statistics.mean(results[results['bucket'] == x]['freq']), decimals=1)
        homoscedastic_ratio = 0
        if np.size(results[results['bucket'] == x]) > 1:
            print("cluster #{0}: \tsum elements:{1}  \tmin value: {2} \tmax value: {3} "
                  "\tmean: {4} \tstd deviation: {5} \tmedian: {6}".format(x,
                                                                          np.size(results[results['bucket'] == x]),
                                                                          np.min(
                                                                              results[results['bucket'] == x]['freq']),
                                                                          np.max(
                                                                              results[results['bucket'] == x]['freq']),
                                                                          mean_bucket,
                                                                          np.around(math.sqrt(np.var(
                                                                              results[results['bucket'] == x]['freq'],
                                                                              ddof=1)), decimals=3),
                                                                          statistics.median(
                                                                              results[results['bucket'] == x]['freq'])
                                                                          )
                  )
            # sv = sv + math.sqrt(np.var(results[results['bucket'] == x]['freq']))  # standard deviation
            # sv = sv + np.std(results[results['bucket'] == x]['freq'])  # standard deviation
            sv = sv + np.var(results[results['bucket'] == x]['freq'], ddof=0)  # variance from population data
        else:
            print("cluster #{0}: \tsum elements:{1}  \tmin value: {2} \tmax value: {3} "
                  "\tmean: {4} \tstd deviation: {5} \tmedian: {6}".format(x,
                                                                          np.size(results[results['bucket'] == x]),
                                                                          np.min(
                                                                              results[results['bucket'] == x]['freq']),
                                                                          np.max(
                                                                              results[results['bucket'] == x]['freq']),
                                                                          mean_bucket,
                                                                          0,
                                                                          statistics.median(
                                                                              results[results['bucket'] == x]['freq'])
                                                                          )
                  )
        '''
        ads.boxplot_std_deviation(results[results['bucket'] == x]['freq'],
                                  "./data/boxplot_bucket_" + str(x) + ".pdf", "boxplot bucket " + str(x), 'distance')
        '''
        # calculate ratio between min and max variance cf. https://www.statisticshowto.datasciencecentral.com/homoscedasticity/
        '''
        Bucket b1 enthält die Daten:
        b1 = [8,7,9,10,6] => somit ist der Mittelwert 8 (40/5)
        
        nun wäre die varianz ja
        s^2 = ((8-8)^2 + (7-8)^2 + ..... + (6-8)^2)/5 => 2
        da ist die varianz nun eine zahl = 2
        
        was ich nicht verstehe, wie kann ich nun von nur einem wert die varianz berechnen?
        meinst du sowas?
        
        (8-8)^2 = 0
        (8-7)^2 = 1
        (8-9)^2 = 1
        (8-10)^2 = 4
        (8-6)^2 = 4
        
        dann wäre die min "varainz" = 0
        und die max "varinaz" = 4
        '''
        var_from_minimum = (mean_bucket - np.min(results[results['bucket'] == x]['freq'])) ** 2
        var_from_maximum = (mean_bucket - np.max(results[results['bucket'] == x]['freq'])) ** 2
        # print("Debug: var from minimum={0}".format(var_from_minimum))
        # print("Debug: var from maximum={0}".format(var_from_maximum))

        if var_from_minimum == 0 and var_from_maximum == 0:
            ratio = 0
        elif var_from_minimum == 0 or var_from_maximum == 0:
            ratio = -1  # cant determine ratio
        elif var_from_minimum == var_from_maximum:
            ratio = 0
        elif var_from_minimum < var_from_maximum:
            ratio = var_from_maximum / var_from_minimum
        elif var_from_minimum > var_from_maximum:
            ratio = var_from_minimum / var_from_maximum

        if 0 <= ratio <= 1.5:
            print("cluster #{0} homoscedasticity given with ratio = {1}".format(x, ratio))
            hom_sum += 1
        else:
            print("cluster #{0} homoscedasticity NOT given with ratio = {1}".format(x, ratio))

    print("homoscedasticity given for {0} of {1} used clusters".format(hom_sum, size))

    sed = np.uint64(1)  # score for distribution in buckets (SDB)
    i = 1
    print("\nnumber of elements: {0}".format(sum(counts_used_buckets)))

    print_classification_legend(results)

    for c in counts_used_buckets:
        # print("bucket #{0} contains {1} elements".format(i, s))
        sed = sed * c
        i += 1
    print("\nscore even distribution (SED) MAX: {:.2e}".format(sed))
    print("score even distribution (SED) MAX: {0}".format(sed))
    value = sed
    digits = 0
    while value > 0:
        value = value // 10
        digits += 1
    print("SED number of digits: {0}\n".format(digits))

    if size <= size_aimed_clusters:
        nuc = size / size_aimed_clusters
        print("number of used clusters (NUC) MAX: {0}".format(nuc))
    else:
        print("size of actually used clusters (={0}) is greater than max allowed clusters (={1}). therefore NUC "
              "could not be calculated".format(size, size_aimed_clusters))

    print("\nsum variances (SV) MIN: {0}".format(sv))
    print("sum variances (SV) MIN: {:.2e}".format(sv))

    digits = 0
    while value > 0:
        value = value // 10
        digits += 1
    print("ratio between SED and sv digits MAX: {0}".format(digits))
    summary = "SUM: NUC(MAX): {0}\t SED(MAX): {1:.2e}\t SV(MIN): {2:.2e}" \
        .format(nuc, sed, sv)
    print(summary)

    # cf https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score
    silhouette_avg = silhouette_score(results['freq'].reshape(-1, 1), results['bucket'])
    print("\nMean Silhouette Coefficient of all clusters MSC(MAX): {0:.2}".format(silhouette_avg))
    print("(best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters)")

    # cf https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html
    davies_bouldin = davies_bouldin_score(results['freq'].reshape(-1, 1), results['bucket'])
    print("\nDavies-Bouldin score DB(MIN): {0:.2}".format(davies_bouldin))
    print("(The minimum score is zero, with lower values indicating better clustering.)")

    if overlapping_ranges(results):
        print("overlapping range found in clusters!!")

    return nuc, sed, sv, silhouette_avg


def overlapping_ranges(results: np.dtype([('bucket', int), ('freq', np.float64)])):
    overlapping_range = False

    clusters = np.unique(results['bucket'])
    dt = np.dtype([('bucket', int), ('freq_min', np.float64), ('freq_max', np.float64)])
    ranges = np.zeros(np.size(clusters), dtype=dt)
    # tmp = np.array([(clusters_array[0], data[data_min_idx])], dtype=dt_result)
    # np.min(results[results['bucket'] == x]['freq']),

    for i, c in enumerate(clusters):
        cluster_min_freq = np.min(results[results['bucket'] == c]['freq'])
        cluster_max_freq = np.max(results[results['bucket'] == c]['freq'])
        tmp = np.array([(c, cluster_min_freq, cluster_max_freq)], dtype=dt)
        ranges[i] = tmp

    ranges_freq_min_sorted = np.sort(ranges, order='freq_min')
    # result_freq_max_sorted = np.sort(result, order='freq_max')

    for i, rs in enumerate(ranges_freq_min_sorted):
        if i != 0:
            if ranges_freq_min_sorted[i - 1]['freq_min'] >= ranges_freq_min_sorted[i]['freq_min']:
                overlapping_range = True

    for i, rs in enumerate(ranges_freq_min_sorted):
        if i != 0:
            if ranges_freq_min_sorted[i - 1]['freq_max'] >= ranges_freq_min_sorted[i]['freq_max']:
                overlapping_range = True

    return overlapping_range


def print_classification_legend(results: np.dtype([('bucket', int), ('freq', np.float64)])):
    clusters = np.unique(results['bucket'])
    dt = np.dtype([('bucket', int), ('freq_min', np.float64), ('freq_max', np.float64), ('elements', int)])
    ranges = np.zeros(np.size(clusters), dtype=dt)

    for i, c in enumerate(clusters):
        cluster_min_freq = np.min(results[results['bucket'] == c]['freq'])
        cluster_max_freq = np.max(results[results['bucket'] == c]['freq'])

        elements = 0
        for result in results:
            if result['bucket'] == c:
                elements += 1

        tmp = np.array([(c, cluster_min_freq, cluster_max_freq, elements)], dtype=dt)
        ranges[i] = tmp

    ranges_freq_min_sorted = np.sort(ranges, order='freq_min')

    print("\nbuilt clusters:")
    for el in ranges_freq_min_sorted:
        print("cluster #{0}: {1} -- {2} ({3})".format(el['bucket'], el['freq_min'], el['freq_max'], el['elements']))


def sort_results_clusters(results: np.dtype([('bucket', int), ('freq', np.float64)])) \
        -> np.dtype([('bucket', int), ('freq', np.float64)]):
    """
    sorting clusters of clustered results asc:
    the minimum frequencies are in cluster 0, and next higher frequencies are in cluster 1, etc.
    :param results: custom numpy of clustered resutls
    :return: sorted clustered results
    """
    clusters = np.unique(results['bucket'])
    dt_ranges = np.dtype([('bucket', int), ('freq_min', np.float64), ('freq_max', np.float64), ('elements', int)])
    ranges = np.zeros(np.size(clusters), dtype=dt_ranges)

    for i, c in enumerate(clusters):
        cluster_min_freq = np.min(results[results['bucket'] == c]['freq'])
        cluster_max_freq = np.max(results[results['bucket'] == c]['freq'])

        elements = 0
        for result in results:
            if result['bucket'] == c:
                elements += 1

        tmp = np.array([(c, cluster_min_freq, cluster_max_freq, elements)], dtype=dt_ranges)
        ranges[i] = tmp

    ranges_freq_min_sorted = np.sort(ranges, order='freq_min')

    dt_result = np.dtype([('bucket', int), ('freq', np.float64)])
    results_sorted = np.zeros(results.size, dtype=dt_result)
    results_sorted_idx = 0

    print("\nbuilt clusters:")
    cluster = 0
    for el in ranges_freq_min_sorted:
        print("cluster #{0}: {1} -- {2} ({3})".format(el['bucket'], el['freq_min'], el['freq_max'], el['elements']))

        for result in results:
            if el['freq_min'] <= result['freq'] <= el['freq_max']:
                tmp = np.array([(cluster, result['freq'])], dtype=dt_result)
                results_sorted[results_sorted_idx] = tmp
                results_sorted_idx += 1
        cluster += 1

    return results_sorted

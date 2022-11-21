# Overview

A heuristic one dimensional clustering algorithm called DDCAL (Density Distribution Cluster Algorithm) that is based on iterative feature scaling.
The algorithm aims as first order to even distribute data into clusters by considering as well as second order to minimize the variance inside each cluster and maximizing the distances between clusters.

The algorithm is designed to be used for visualization, e.g., on choropleth maps.


# Basic Usage
```
pip install -i https://pypi.org/simple/ ddcal
```

```
from clustering.ddcal import DDCAL

# load data
frequencies = [0, 1, 1, 1, 5, 5, 5, 30, 88]

# initialize parameters
ddcal = DDCAL(n_clusters=3, feature_boundary_min=0.1, feature_boundary_max=0.49,
                  num_simulations=20, q_tolerance=0.45, q_tolerance_increase_step=0.5)

# execute DDCAL algorithm
ddcal.fit(frequencies)

# print/use results
print(ddcal.sorted_data)
print(ddcal.labels_sorted_data)
```

# Supplemental Material

Supplemental material for the paper **DDCAL: Evenly Distributing Data into High Density Clusters based on Iterative Feature Scaling** can be found in the folder:

```
supplemental
```

# Synthetic Data Sets

The synthetic data sets, which were used in the paper **DDCAL: Evenly Distributing Data into High Density Clusters based on Iterative Feature Scaling** which includes a description on each data set, can be found in the folder:

```
tests/data
```

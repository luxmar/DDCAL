# Synthetic data sets for testing DDCAL with different distributions

- all data sets contain 1000 elements
- on all created data sets, each element was rounded to two decimals to have non-unique elements  
```
size = 1000
```

# normal distribution
```
mu = 0.1
sigma = 1
```

created with
```
numpy.random.normal(mu, sigma, size)
```

about data
```
mean = 0.50271
median = 0.52
min value = -2.77
max value = 4.53
stdandard deviation = 0.9831733091881614
unique data points = 369
```

# gumbel distribution
```
mu = 0.1
sigma = 1
```

created with
```
numpy.random.gumbel(mu, sigma, size)
```

about data
```
mean = 1.03117
median = 0.85
min value = -1.48
max value = 6.96
stdandard deviation = 1.168495926864959
unique data points = 393
```

# uniform distribution

created with
```
numpy.random.uniform(low=-4, high=4, size)
```

about data
```
mean = -0.08256
median = -0.12
min value = -3.99
max value = 4.0
stdandard deviation = 2.3230632463193936
unique data points = 558
```

# exponential distribution

created with
```
numpy.random.exponential(scale=4, size)
```

about data
```
mean = 3.89401
median = 2.68
min value = 0.0
max value = 26.46
stdandard deviation = 3.8567273458075824
unique data points = 587
```

# two peaks distribution
```
divider = 10
```

created with
```
hstack((numpy.random.normal(loc=-2.5, scale=0.5, size=int(size/divider*3)), numpy.random.normal(loc=2.5, scale=0.5, size=int(size/divider*7))))
```

about data
```
mean = 1.01941
median = 2.24
min value = -3.58
max value = 3.85
stdandard deviation = 2.3237137413846827
unique data points = 362
```

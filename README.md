Fast k-Nearest Neighbor Classifier
================

-   [Fast Nearest Neighbour Searching](#fast-nearest-neighbour-searching)

> Fast KNN with shrinkage estimator for the class membership probabilities

Fast Nearest Neighbour Searching
--------------------------------

The `fastknn` method implements a k-Nearest Neighbor (KNN) classifier based on the [ANN](https://www.cs.umd.edu/~mount/ANN/) library. ANN is written in `C++` and is able to find the k nearest neighbors for every point in a given dataset in `O(N log N)` time.

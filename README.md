Fast k-Nearest Neighbor Classifier
================

-   [Fast Nearest Neighbor Searching](#fast-nearest-neighbor-searching)
-   [The FastKNN Classifier](#the-fastknn-classifier)
-   [Find the Best k](#find-the-best-k)
-   [Plot Classification Decision Boundary](#plot-classification-decision-boundary)
-   [Benchmark](#benchmark)

> Fast KNN with shrinkage estimator for the class membership probabilities

Fast Nearest Neighbor Searching
-------------------------------

The `fastknn` method implements a k-Nearest Neighbor (KNN) classifier based on the [ANN](https://www.cs.umd.edu/~mount/ANN) library. ANN is written in `C++` and is able to find the k nearest neighbors for every point in a given dataset in `O(N log N)` time. The package [RANN](https://github.com/jefferis/RANN) provides an easy interface to use ANN library in `R`.

The FastKNN Classifier
----------------------

The `fastknn` was developed to deal with very large datasets (&gt; 100k rows) and is ideal to [Kaggle](https://www.kaggle.com) competitions. It can be about 50x faster then the popular `knn` method from the `R` package [class](https://cran.r-project.org/web/packages/class), for large datasets. Moreover, `fastknn` provides a shrinkage estimator to the class membership probabilities, based on the inverse distances of the nearest neighbors (**see the PDF version**):

\[
P(x_i \in y_j) = \displaystyle\frac{\displaystyle\sum\limits_{k=1}^K \left( \frac{1}{d_{ik}}\cdot(n_{ik} \in y_j) \right)}{\displaystyle\sum\limits_{k=1}^K \left( \frac{1}{d_{ik}} \right)}
\]

where \(x_i\) is the \(i^{\text{th}}\) test instance, \(y_j\) is the \(j^{\text{th}}\) unique class label, \(n_{ik}\) is the \(k^{\text{th}}\) nearest neighbor of \(x_i\), and \(d_{ik}\) is the distance between \(x_i\) and \(n_{ik}\). This estimator can be thought of as a weighted voting rule, where those neighbors that are more close to \(x_i\) will have more influence on predicting \(x_i\)'s label.

In general, the weighted estimator provides more **calibrated probabilities** when compared with the traditional estimator based on the label proportions of the nearest neighbors, and reduces **logarithmic loss** (log-loss).

### How to install `fastknn`?

The package `fastknn` is not on CRAN, so you need to install it directly from GitHub:

``` r
library("devtools")
install_github("davpinto/fastknn")
```

### Required Packages

The base of `fastknn` is the `RANN` package, but other packages are required to make `fastknn` work properly. All of them are automatically installed when you install the `fastknn`.

-   `RANN` for fast nearest neighbors searching,
-   `magrittr` to use the pipe operator `%>%`,
-   `Metrics` to measure classification performance,
-   `ggplot2` to plot classification decision boundaries,
-   `viridis` for modern color palletes.

### Getting Started

Using `fastknn` is as simple as:

``` r
## Load packages
library("fastknn")
library("caTools")

## Load toy data
data("chess", package = "fastknn")

## Split data for training and test
tr.idx <- caTools::sample.split(Y = chess$y, SplitRatio = 0.7)
x.tr   <- chess$x[tr.idx, ]
x.te   <- chess$x[-tr.idx, ]
y.tr   <- chess$y[tr.idx]
y.te   <- chess$y[-tr.idx]

## Fit KNN
yhat <- fastknn(x.tr, y.tr, x.te, k = 10)

## Evaluate model on test set
sprintf("Accuracy: %.2f", 100 * sum(yhat$class == y.te) / length(y.te))
```

    ## [1] "Accuracy: 99.60"

Find the Best k
---------------

The `fastknn` provides a interface to select the best `k` using n-fold cross-validation. There 4 possible **loss functions**:

-   Overall classification error rate: `eval.metric = "overall_error"`
-   Mean in-class classification error rate: `eval.metric = "mean_error"`
-   Mean in-class AUC: `eval.metric = "auc"`
-   Cross-entropy / logarithmic loss: `eval.metric = "logloss"`

``` r
cv.out <- fastknnCV(chess$x, chess$y, k = 3:10, folds = 5, eval.metric = "logloss")
cv.out$cv_table
```

<table style="width:78%;">
<colgroup>
<col width="12%" />
<col width="12%" />
<col width="12%" />
<col width="12%" />
<col width="12%" />
<col width="11%" />
<col width="4%" />
</colgroup>
<thead>
<tr class="header">
<th align="center">fold_1</th>
<th align="center">fold_2</th>
<th align="center">fold_3</th>
<th align="center">fold_4</th>
<th align="center">fold_5</th>
<th align="center">mean</th>
<th align="center">k</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center">0.1903</td>
<td align="center">0.1079</td>
<td align="center">0.1156</td>
<td align="center">0.1021</td>
<td align="center">0.1108</td>
<td align="center">0.1253</td>
<td align="center">3</td>
</tr>
<tr class="even">
<td align="center">0.1131</td>
<td align="center">0.0283</td>
<td align="center">0.1158</td>
<td align="center">0.1056</td>
<td align="center">0.03456</td>
<td align="center">0.07948</td>
<td align="center">4</td>
</tr>
<tr class="odd">
<td align="center">0.03367</td>
<td align="center">0.02878</td>
<td align="center">0.119</td>
<td align="center">0.03051</td>
<td align="center">0.03519</td>
<td align="center">0.04943</td>
<td align="center">5</td>
</tr>
<tr class="even">
<td align="center">0.03087</td>
<td align="center">0.03271</td>
<td align="center">0.1206</td>
<td align="center">0.03395</td>
<td align="center">0.0356</td>
<td align="center">0.05075</td>
<td align="center">6</td>
</tr>
<tr class="odd">
<td align="center">0.03544</td>
<td align="center">0.03605</td>
<td align="center">0.123</td>
<td align="center">0.03548</td>
<td align="center">0.03683</td>
<td align="center">0.05335</td>
<td align="center">7</td>
</tr>
<tr class="even">
<td align="center">0.03736</td>
<td align="center">0.03617</td>
<td align="center">0.04127</td>
<td align="center">0.03533</td>
<td align="center">0.03749</td>
<td align="center">0.03752</td>
<td align="center">8</td>
</tr>
<tr class="odd">
<td align="center">0.0401</td>
<td align="center">0.04094</td>
<td align="center">0.04237</td>
<td align="center">0.03635</td>
<td align="center">0.03794</td>
<td align="center">0.03954</td>
<td align="center">9</td>
</tr>
<tr class="even">
<td align="center">0.04242</td>
<td align="center">0.04259</td>
<td align="center">0.04379</td>
<td align="center">0.03722</td>
<td align="center">0.03913</td>
<td align="center">0.04103</td>
<td align="center">10</td>
</tr>
</tbody>
</table>

Plot Classification Decision Boundary
-------------------------------------

Benchmark
---------

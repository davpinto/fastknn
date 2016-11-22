library("fastknn")
library("caTools")

context("Fit and predict")

#### Prepare dataset example
dt <- iris
x  <- data.matrix(dt[, -5])
y  <- iris$Species
tr.idx <- caTools::sample.split(Y = y, SplitRatio = 0.7)
x.tr <- x[tr.idx,]
x.te <- x[-tr.idx,]
y.tr <- y[tr.idx]

test_that("if method vote predicts class labels", {
   knn.out <- fastknn(xtr = x.tr, ytr = y.tr, xte = x.te, k = 5, method = "vote")
   expect_is(knn.out$class, "factor")
   expect_length(knn.out$class, nrow(x.te))
   expect_equal(nlevels(knn.out$class), nlevels(y.tr))
   expect_equal(levels(knn.out$class), levels(y.tr))
})

test_that("if method vote predicts class probabilities", {
   knn.out <- fastknn(xtr = x.tr, ytr = y.tr, xte = x.te, k = 5, method = "vote")
   expect_is(knn.out$prob, "matrix")
   expect_equal(nrow(knn.out$prob), nrow(x.te))
   expect_equal(ncol(knn.out$prob), nlevels(y.tr))
   expect_equal(colnames(knn.out$prob), levels(y.tr))
   expect_true(all(knn.out$prob <= 1))
   expect_true(all(knn.out$prob >= 0))
   expect_true(all(rowSums(knn.out$prob) <= 1))
})

test_that("if method dist predicts class labels", {
   knn.out <- fastknn(xtr = x.tr, ytr = y.tr, xte = x.te, k = 5, method = "dist")
   expect_is(knn.out$class, "factor")
   expect_length(knn.out$class, nrow(x.te))
   expect_equal(nlevels(knn.out$class), nlevels(y.tr))
   expect_equal(levels(knn.out$class), levels(y.tr))
})

test_that("if method dist predicts class probabilities", {
   knn.out <- fastknn(xtr = x.tr, ytr = y.tr, xte = x.te, k = 5, method = "dist")
   expect_is(knn.out$prob, "matrix")
   expect_equal(nrow(knn.out$prob), nrow(x.te))
   expect_equal(ncol(knn.out$prob), nlevels(y.tr))
   expect_equal(colnames(knn.out$prob), levels(y.tr))
   expect_true(all(knn.out$prob <= 1))
   expect_true(all(knn.out$prob >= 0))
   expect_true(all(rowSums(knn.out$prob) <= 1))
})

# context("Cross validation")
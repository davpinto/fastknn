pbapply::pboptions(type = "none")

#### Data example
x.tr <- x[tr.ids,]
x.te <- x[-tr.ids,]
y.tr <- y[tr.ids]
y.te <- y[-tr.ids]

context("Normalize data")

test_that("data standardization", {
   x.sc <- fastknn:::scaleData(x.tr, x.te, type = "std")
   x.mu <- rep(0, ncol(x.tr))
   names(x.mu) <- colnames(x.tr)
   x.sd <- rep(1, ncol(x.tr))
   names(x.sd) <- colnames(x.tr)
   expect_equal(dim(x.tr), dim(x.sc$new.tr))
   expect_equal(dim(x.te), dim(x.sc$new.te))
   expect_equal(colMeans(x.sc$new.tr), x.mu, tolerance = 1e-15)
   expect_equal(apply(x.sc$new.tr, 2, sd), x.sd, tolerance = 1e-15)
   expect_equivalent(x.sc$new.te, scale(x.te, center = apply(x.tr, 2, mean), 
                                        scale = apply(x.tr, 2, sd)))
})

test_that("min_max normalization", {
   x.sc <- fastknn:::scaleData(x.tr, x.te, type = "minmax")
   x.min <- rep(0, ncol(x.tr))
   names(x.min) <- colnames(x.tr)
   x.max <- rep(1, ncol(x.tr))
   names(x.max) <- colnames(x.tr)
   expect_equal(dim(x.tr), dim(x.sc$new.tr))
   expect_equal(dim(x.te), dim(x.sc$new.te))
   expect_equal(apply(x.sc$new.tr, 2, min), x.min, tolerance = 1e-15)
   expect_equal(apply(x.sc$new.tr, 2, max), x.max, tolerance = 1e-15)
   expect_equivalent(x.sc$new.te, scale(x.te, center = apply(x.tr, 2, min), 
                                        scale = apply(x.tr, 2, max) - 
                                           apply(x.tr, 2, min)))
})

test_that("max_abs scaling", {
   x.sc <- fastknn:::scaleData(x.tr, x.te, type = "maxabs")
   x.max <- rep(1, ncol(x.tr))
   names(x.max) <- colnames(x.tr)
   expect_equal(dim(x.tr), dim(x.sc$new.tr))
   expect_equal(dim(x.te), dim(x.sc$new.te))
   expect_equal(apply(x.sc$new.tr, 2, max), x.max, tolerance = 1e-15)
   expect_true(all(apply(x.sc$new.tr, 2, min) >= -1))
})

test_that("robust normalization", {
   x.sc <- fastknn:::scaleData(x.tr, x.te, type = "robust")
   x.mu <- apply(x.tr, 2, median)
   names(x.mu) <- colnames(x.tr)
   x.sd <- apply(x.tr, 2, IQR)
   names(x.sd) <- colnames(x.tr)
   expect_equal(dim(x.tr), dim(x.sc$new.tr))
   expect_equal(dim(x.te), dim(x.sc$new.te))
   expect_equivalent(x.sc$new.tr, scale(x.tr, center = x.mu, scale = x.sd))
   expect_equivalent(x.sc$new.te, scale(x.te, center = x.mu, scale = x.sd))
})

context("Fit and predict")

test_that("normalization method not found", {
   expect_error(fastknn(x.tr, y.tr, x.te, k = 5, normalize = 'none'))
})

test_that("if method vote predicts class labels", {
   knn.out <- fastknn(xtr = x.tr, ytr = y.tr, xte = x.te, k = 5, 
                      method = "vote", normalize = "maxabs")
   expect_is(knn.out$class, "factor")
   expect_length(knn.out$class, nrow(x.te))
   expect_equal(nlevels(knn.out$class), nlevels(y.tr))
   expect_equal(levels(knn.out$class), levels(y.tr))
})

test_that("if method vote predicts class probabilities", {
   knn.out <- fastknn(xtr = x.tr, ytr = y.tr, xte = x.te, k = 5, 
                      method = "vote", normalize = "maxabs")
   prob.sums <- rep(1, nrow(x.te))
   expect_is(knn.out$prob, "matrix")
   expect_equal(nrow(knn.out$prob), nrow(x.te))
   expect_equal(ncol(knn.out$prob), nlevels(y.tr))
   expect_equal(colnames(knn.out$prob), levels(y.tr))
   expect_true(all(knn.out$prob <= 1))
   expect_true(all(knn.out$prob >= 0))
   expect_equal(rowSums(knn.out$prob), prob.sums, tolerance = 1e-15)
})

test_that("if method dist predicts class labels", {
   knn.out <- fastknn(xtr = x.tr, ytr = y.tr, xte = x.te, k = 5, 
                      method = "dist", normalize = "maxabs")
   expect_is(knn.out$class, "factor")
   expect_length(knn.out$class, nrow(x.te))
   expect_equal(nlevels(knn.out$class), nlevels(y.tr))
   expect_equal(levels(knn.out$class), levels(y.tr))
})

test_that("if method dist predicts class probabilities", {
   knn.out <- fastknn(xtr = x.tr, ytr = y.tr, xte = x.te, k = 5, 
                      method = "dist", normalize = "maxabs")
   prob.sums <- rep(1, nrow(x.te))
   expect_is(knn.out$prob, "matrix")
   expect_equal(nrow(knn.out$prob), nrow(x.te))
   expect_equal(ncol(knn.out$prob), nlevels(y.tr))
   expect_equal(colnames(knn.out$prob), levels(y.tr))
   expect_true(all(knn.out$prob <= 1))
   expect_true(all(knn.out$prob >= 0))
   expect_true(all(rowSums(knn.out$prob) == 1))
   expect_equal(rowSums(knn.out$prob), prob.sums, tolerance = 1e-15)
})

context("Parameter tuning")

test_that("stratified fold sampling", {
   fold.ids <- fastknn:::createCVFolds(y.tr, n = 5)
   expect_is(fold.ids, "integer")
   expect_length(unique(fold.ids), 5)
   expect_equal(as.numeric(prop.table(table(y.tr[fold.ids == 1]))),
                as.numeric(prop.table(table(y.tr))))
   
})

test_that("n-fold cross validation", {
   # https://github.com/hadley/testthat/issues/129
   Sys.setenv("R_TESTS" = "")
   
   k.list <- 1:3
   nfolds <- 3
   cv.out <- fastknnCV(x.tr, y.tr, k = k.list, folds = nfolds)
   expect_equal(ncol(cv.out$cv_table), nfolds + 2)
   expect_equal(nrow(cv.out$cv_table), length(k.list))
   expect_equal(cv.out$best_k, cv.out$cv_table$k[which.min(cv.out$cv_table$mean)])
   expect_equal(cv.out$best_eval, min(cv.out$cv_table$mean))
})

test_that("parallelized n-fold cross validation", {
   Sys.setenv("R_TESTS" = "")
   k.list <- 1:3
   nfolds <- 3
   cv.out <- fastknnCV(x.tr, y.tr, k = k.list, folds = nfolds, nthread = 2)
   expect_equal(ncol(cv.out$cv_table), nfolds + 2)
   expect_equal(nrow(cv.out$cv_table), length(k.list))
   expect_equal(cv.out$best_k, cv.out$cv_table$k[which.min(cv.out$cv_table$mean)])
   expect_equal(cv.out$best_eval, min(cv.out$cv_table$mean))
})

test_that("n-fold cross validation with pre-defined ids", {
   Sys.setenv("R_TESTS" = "")
   k.list <- 1:3
   cv.out <- fastknnCV(x, y, k = k.list, folds = cv.ids)
   expect_equal(ncol(cv.out$cv_table), length(unique(cv.ids)) + 2)
   expect_equal(nrow(cv.out$cv_table), length(k.list))
   expect_equal(cv.out$best_k, cv.out$cv_table$k[which.min(cv.out$cv_table$mean)])
   expect_equal(cv.out$best_eval, min(cv.out$cv_table$mean))
})

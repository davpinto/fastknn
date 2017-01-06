pbapply::pboptions(type = "none")

#### Data example
x.tr <- x[tr.ids,]
x.te <- x[-tr.ids,]
y.tr <- y[tr.ids]
y.te <- y[-tr.ids]

context("Extract features")

test_that("Feature extraction", {
   Sys.setenv("R_TESTS" = "")
   
   nfolds <- 3
   k <- 3
   dt <- knnExtract(x.tr, y.tr, x.te, k = k, folds = nfolds)
   expect_is(dt$new.tr, "matrix")
   expect_is(dt$new.te, "matrix")
   expect_equal(nrow(dt$new.tr), nrow(x.tr))
   expect_equal(nrow(dt$new.te), nrow(x.te))
   expect_equal(ncol(dt$new.tr), k * nlevels(y.tr))
   expect_equal(ncol(dt$new.te), k * nlevels(y.tr))
})

test_that("Parallelized feature extraction", {
   Sys.setenv("R_TESTS" = "")
   
   nfolds <- 3
   k <- 3
   dt <- knnExtract(x.tr, y.tr, x.te, k = k, folds = nfolds, nthread = 2)
   expect_is(dt$new.tr, "matrix")
   expect_is(dt$new.te, "matrix")
   expect_equal(nrow(dt$new.tr), nrow(x.tr))
   expect_equal(nrow(dt$new.te), nrow(x.te))
   expect_equal(ncol(dt$new.tr), k * nlevels(y.tr))
   expect_equal(ncol(dt$new.te), k * nlevels(y.tr))
})

test_that("Feature extraction with pre-defined fold ids", {
   Sys.setenv("R_TESTS" = "")
   
   k <- 3
   dt <- knnExtract(x, y, x.te, k = k, folds = cv.ids)
   expect_is(dt$new.tr, "matrix")
   expect_is(dt$new.te, "matrix")
   expect_equal(nrow(dt$new.tr), nrow(x))
   expect_equal(nrow(dt$new.te), nrow(x.te))
   expect_equal(ncol(dt$new.tr), k * nlevels(y.tr))
   expect_equal(ncol(dt$new.te), k * nlevels(y.tr))
})

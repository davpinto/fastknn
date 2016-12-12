#### Data example
x.tr <- x[tr.ids,]
x.te <- x[-tr.ids,]
y.tr <- y[tr.ids]
y.te <- y[-tr.ids]

#### Fit KNN
knn.out <- fastknn(x.tr, y.tr, x.te, k = 10)

context("Compute classification loss")

test_that("Overall classification loss", {
   knn.loss <- classLoss(y.te, knn.out$class, eval.metric = "overall_error")
   expect_is(knn.loss, "numeric")
   expect_true(is.finite(knn.loss))
   expect_lte(knn.loss, 1)
   expect_gte(knn.loss, 0)
})

test_that("Mean per-class classification loss", {
   knn.loss <- classLoss(y.te, knn.out$class, eval.metric = "mean_error")
   expect_is(knn.loss, "numeric")
   expect_true(is.finite(knn.loss))
   expect_lte(knn.loss, 1)
   expect_gte(knn.loss, 0)
})

test_that("Mean per-class AUC", {
   knn.loss <- classLoss(y.te, prob = knn.out$prob, eval.metric = "auc")
   expect_is(knn.loss, "numeric")
   expect_true(is.finite(knn.loss))
   expect_lte(knn.loss, 1)
   expect_gte(knn.loss, 0.5)
})

test_that("Cross-entropy loss", {
   knn.loss <- classLoss(y.te, prob = knn.out$prob, eval.metric = "logloss")
   expect_is(knn.loss, "numeric")
   expect_true(is.finite(knn.loss))
   expect_gte(knn.loss, 0)
})

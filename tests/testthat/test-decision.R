#### Data example
x.tr <- x[tr.ids,]
x.te <- x[-tr.ids,]
y.tr <- y[tr.ids]
y.te <- y[-tr.ids]

context("Plot classification decision")

test_that("Plot only training data", {
   g <- knnDecision(x.tr, y.tr, k = 10)
   expect_true(ggplot2::is.ggplot(g))
   expect_is(ggplot2::ggplot_build(g), "list")
})

test_that("Plot with test data", {
   g <- knnDecision(x.tr, y.tr, x.te, y.te, k = 10)
   expect_true(ggplot2::is.ggplot(g))
   expect_is(ggplot2::ggplot_build(g), "list")
})
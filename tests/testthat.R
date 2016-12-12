library("testthat")
library("fastknn")

#### Create internal data
# x <- data.matrix(iris[,-5])
# y <- iris$Species
# tr.ids <- caret::createDataPartition(y, p = 0.7, list = FALSE)[,1]
# cv.ids <- caret::createFolds(y, k = 5, list = FALSE)
# devtools::use_data(x,y,tr.ids,cv.ids,internal = TRUE, overwrite = TRUE)

test_check("fastknn")

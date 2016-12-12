#' Fast KNN Classifier
#'
#' Fast k-Nearest Neighbor classifier build upon ANN, a high efficient 
#' \code{C++} library for nearest neighbor searching.
#' 
#' There are two estimators for the class membership probabilities:
#' \enumerate{
#' \item \code{method="vote"}: The classical estimator based on the label 
#' proportions of the nearest neighbors. This estimator can be thought as of a 
#' \strong{voting} rule.
#' \item  \code{method="dist"}: A shrinkage estimator based on the distances 
#' from the nearest neighbors, so that those neighbors more close to the test 
#' observation have more importance on predicting the class label. This 
#' estimator can be thought as of a \strong{weighted voting} rule. In general, 
#' it reduces log-loss.
#' }
#' 
#' @param xtr matrix containing the training instances. Rows are observations 
#' and columns are variables. Only numeric variables are allowed.
#' @param xte matrix containing the test instances.
#' @param ytr factor array with the training labels.
#' @param k number of neighbors considered.
#' @param method method used to infer the class membership probabilities of the 
#' test instances. Choose \code{"dist"} (default) to compute probabilites from 
#' the inverse of the nearest neighbor distances. This method works as
#' a shrinkage estimator and provides a better predictive performance in general.
#' Or you can choose \code{"vote"} to compute probabilities from the frequency 
#' of the nearest neighbor labels.
#' @param normalize variable normalization to be applied prior to searching the 
#' nearest neighbors. Default is \code{normalize=NULL}. Normalization is 
#' recommended if variables are not in the same units. It can be one of the 
#' following:
#' \itemize{
#'  \item{normalize="std"}: standardize variables by removing the mean and 
#'  scaling to unit variance. 
#'  \item{normalize="minmax"}: transforms variables by scaling each one between 
#'  0 and 1.
#'  \item{normalize="maxabs"}: scales each variable by its maximum absolute 
#'  value. This is the best choice for sparse data because it does not 
#'  shift/center the variables.
#'  \item{normalize="robust"}: scales variables using statistics that are 
#'  robust to outliers. It removes the median and scales by the interquartile 
#'  range (IQR).
#' }
#'
#' @return \code{list} with predictions for the test set:
#' \itemize{
#'  \item \code{class}: factor array of predicted classes.
#'  \item \code{prob}: matrix with predicted probabilities.
#' }
#' 
#' @author 
#' David Pinto.
#' 
#' @export
#' 
#' @examples
#' \dontrun{
#' library("mlbench")
#' library("caTools")
#' library("fastknn")
#' 
#' data("Ionosphere")
#' 
#' x <- data.matrix(subset(Ionosphere, select = -Class))
#' y <- Ionosphere$Class
#' 
#' set.seed(2048)
#' tr.idx <- which(sample.split(Y = y, SplitRatio = 0.7))
#' x.tr <- x[tr.idx,]
#' x.te <- x[-tr.idx,]
#' y.tr <- y[tr.idx]
#' y.te <- y[-tr.idx]
#' 
#' knn.out <- fastknn(xtr = x.tr, ytr = y.tr, xte = x.te, k = 10)
#' 
#' knn.out$class
#' knn.out$prob
#' }
fastknn <- function(xtr, ytr, xte, k, method = "dist", normalize = NULL) {
   
   #### Check args
   checkKnnArgs(xtr, ytr, xte, k)
   stopifnot(method %in% c("vote","dist"))
   
   #### Normalize data
   if (!is.null(normalize)) {
      norm.out <- scaleData(xtr, xte, type = normalize)
      xtr <- norm.out$new.tr
      xte <- norm.out$new.te
      rm("norm.out")
      gc()
   }
   
   #### Find nearest neighbors
   knn.search <- RANN::nn2(data = xtr, query = xte, k = k, treetype = 'kd', 
                           searchtype = 'standard')
   
   #### Compute class membership probabilities
   label.mat <- matrix(ytr[knn.search$nn.idx], ncol = k)
   knn.prob <- switch(
      method,
      ## P(y_j | x_i) = sum(1/d(nn_i) * (y(nn_i) == y_j)) / sum(1/d(nn_i))
      'dist' = {
         sapply(levels(ytr), function(cl, d, y) {
            rowSums(1/d * (y == cl)) / rowSums(1/d)
         }, d = pmax(knn.search$nn.dists, 1e-15), y = label.mat, 
         simplify=FALSE, USE.NAMES=TRUE)
      },
      ## P(y_j | x_i) = sum(y(nn_i) == y_j) / k
      'vote' = {
         sapply(levels(ytr), function(cl, y) {
            rowSums(y == cl) / ncol(y)
         }, y = label.mat, simplify=FALSE, USE.NAMES=TRUE)
      }
   )
   knn.prob <- as.matrix(do.call('cbind.data.frame', knn.prob))
   knn.prob <- sweep(knn.prob, 1, rowSums(knn.prob), "/")
   rm(list = c('knn.search', 'label.mat'))
   gc()
   
   #### Assign class labels
   knn.label <- levels(ytr)[max.col(knn.prob, ties.method = "first")]
   knn.label <- factor(knn.label, levels(ytr))
   
   return(list(
      class = knn.label,
      prob = knn.prob
   ))
}

#' Cross-Validation for fastknn
#'
#' Does n-fold cross-validation for \code{fastknn} to find the best k parameter.
#'
#' @param x input matrix of dimension \code{nobs x nvars}.
#' @param y factor array wtih class labels for the \code{x} rows.
#' @param k sequence of possible k values to be evaluated (default is [3:15]).
#' @param method the probability estimator as in \code{\link{fastknn}}.
#' @param normalize variable scaler as in \code{\link{fastknn}}.
#' @param folds number of folds (default is 5) or an array with fold ids between 
#' 1 and \code{n} identifying what fold each observation is in. The smallest 
#' value allowable is \code{nfolds=3}. The fold assigment given by 
#' \code{fastknnCV} does stratified sampling.
#' @param eval.metric classification loss measure to use in cross-validation. 
#' See \code{\link{classLoss}} for more details.
#' 
#' @return \code{list} with cross-validation results:
#' \itemize{
#'  \item \code{best_eval}: the best loss measure found in the 
#'  cross-validation procedure.
#'  \item \code{best_k}: the best k value found in the cross-validation procedure.
#'  \item \code{cv_table}: \code{data.frame} with the test performances for each k 
#'  on each data fold. 
#' }
#' 
#' @author 
#' David Pinto.
#' 
#' @seealso 
#' \code{\link{classLoss}}
#' 
#' @export
#' 
#' @examples
#' \dontrun{
#' library("mlbench")
#' library("caTools")
#' library("fastknn")
#' 
#' data("Ionosphere")
#' 
#' x <- data.matrix(subset(Ionosphere, select = -Class))
#' y <- Ionosphere$Class
#' 
#' set.seed(1024)
#' tr.idx <- which(sample.split(Y = y, SplitRatio = 0.7))
#' x.tr <- x[tr.idx,]
#' x.te <- x[-tr.idx,]
#' y.tr <- y[tr.idx]
#' y.te <- y[-tr.idx]
#' 
#' set.seed(2048)
#' cv.out <- fastknnCV(x = x.tr, y = y.tr, k = c(5,10,15,20), eval.metric="logloss")
#' 
#' cv.out$cv_table
#' }
fastknnCV <- function(x, y, k = 3:15, method = "dist", normalize = NULL, 
                      folds = 5, eval.metric = "overall_error") {
   
   #### Check and create data folds
   if (length(folds) > 1) {
      if (length(unique(folds)) < 3) {
         stop('The smallest number of folds allowable is 3')
      }
      if (length(unique(folds)) > nrow(x)) {
         stop('The highest number of folds allowable is nobs (leave-one-out CV)')
      }
   } else {
      folds <- min(max(3, folds), nrow(x))
      folds <- createCVFolds(y, n = folds)
   }
   
   #### n-fold cross validation
   folds <- factor(paste('fold', folds, sep = '_'), 
                   levels = paste('fold', sort(unique(folds)), sep = '_')) 
   cv.results <- pbapply::pblapply(k, function(k, x, y, folds) {
      sapply(levels(folds), function(fold.id) {
         te.idx <- which(folds == fold.id)
         y.hat <- fastknn(x[-te.idx,], y[-te.idx], x[te.idx,], k, method, 
                          normalize)
         classLoss(actual = y[te.idx], predicted = y.hat$class, 
                   prob = y.hat$prob, eval.metric = eval.metric)
      }, simplify = FALSE, USE.NAMES = TRUE)
   }, x = x, y = y, folds = folds)
   cv.results <- do.call('rbind.data.frame', cv.results)
   cv.results$mean <- rowMeans(cv.results)
   cv.results$k <- k
   
   #### Select best performance
   if (eval.metric == "auc") {
      best.idx <- which.max(cv.results$mean)
   } else {
      best.idx <- which.min(cv.results$mean)
   }
   
   return(list(
      best_k = k[best.idx],
      best_eval = cv.results$mean[best.idx],
      cv_table = cv.results
   ))
}

#### Scale features
scaleData <- function(xtr, xte, type = "maxabs") {
   stopifnot(type %in% c("std", "minmax", "maxabs", "robust"))
   
   ## Compute column center and scale
   switch(
      type,
      std = { # Data standardization
         x.center <- colMeans(xtr)
         x.scaler <- matrixStats::colSds(xtr)
      },
      minmax = { # Normalize between 0 and 1
         x.center <- matrixStats::colMins(xtr)
         x.scaler <- matrixStats::colMaxs(xtr) - x.center
      },
      maxabs = { # Set max value as 1 and keep sparsity
         x.center <- rep(0, ncol(xtr))
         x.scaler <- matrixStats::colMaxs(abs(xtr))
      },
      robust = { # Robust to outliers
         x.center <- matrixStats::colMedians(xtr)
         x.scaler <- matrixStats::colIQRs(xtr)
      }
   )
   
   ## Apply normalization
   x.scaler[x.scaler == 0] <- 1
   xtr <- sweep(xtr, 2, x.center, "-")
   xtr <- sweep(xtr, 2, x.scaler, "/")
   xte <- sweep(xte, 2, x.center, "-")
   xte <- sweep(xte, 2, x.scaler, "/")
   
   return(list(
      new.tr = xtr,
      new.te = xte
   ))
}

#### Split data into folds using stratified sampling
createCVFolds <- function(y, n) {
   stopifnot(is.factor(y))
   
   folds <- integer(length(y))
   for (i in levels(y)) {
      folds[which(y == i)] <- sample(cut(1:sum(y == i), breaks = n, labels = FALSE))
   }
   
   return(folds)
}

#### Validate fastknn() parameters
checkKnnArgs <- function(xtr, ytr, xte, k) {
   stopifnot(is.matrix(xtr))
   stopifnot(is.factor(ytr))
   stopifnot(nrow(xtr) == length(ytr))
   stopifnot(is.matrix(xte))
   stopifnot(ncol(xtr) == ncol(xte))
   stopifnot(is.numeric(k))
   if (nlevels(ytr) < 2) {
      stop("Data must contain at least 2 class labels.")
   }
   if (length(k) > 1) {
      stop("k must be a single value.")
   }
   if (k > nrow(xtr)) {
      stop("The number of nearest neighbors cannot be greater than the number of training instances.")
   }
   if (k < 1) {
      stop("k must be at least 1.")
   }
}
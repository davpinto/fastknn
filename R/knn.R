#' Fast KNN Classifier
#'
#' Fast k-Nearest Neighbor classifier build upon ANN, a high efficient 
#' \code{C++} library for nearest neighbor searching.
#'
#' @param xtr matrix containing the training instances.
#' @param xte matrix containing the test instances.
#' @param ytr factor array with the training labels.
#' @param k number of neighbors considered.
#' @param method method used to infer the class membership probabilities of the 
#' test instances. Choose \code{"dist"} (default) to compute probabilites from 
#' the inverse of the nearest neighbor distances. This method works as
#' a shrinkage estimator and provides a better predictive performance in general.
#' Or you can choose \code{"vote"} to compute probabilities from the frequency 
#' of the nearest neighbor labels.
#'
#' @return \code{list} with predictions for the test set:
#' \itemize{
#'  \item \code{class}: factor array of predicted classes.
#'  \item \code{prob}: matrix with predicted probabilities.
#' }
#'
#' @export
fastknn <- function(xtr, ytr, xte, k, method = "dist") {
   
   #### Check args
   if (length(k) > 1) {
      stop("k must be a single value")
   }
   if (k > nrow(xtr)) {
      stop("The number of nearest neighbors cannot be greater than the number of training instances.")
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
#' @param method the probability estimator as in \code{fastknn}.
#' @param folds number of folds (default is 5) or an array with fold ids between 
#' 1 and \code{n} identifying what fold each observation is in. The fold 
#' assigment given by \code{fastknnCV} does stratified sampling.
#' @param eval loss to use for cross-validation. Currently five options are available:
#' \itemize{
#'  \item \code{eval.metric="overal_error"}: default option. It gives the overall 
#'  misclassification rate.
#'  \item \code{eval.metric="mean_error"}: gives the average in-class 
#'  misclassification rate.
#'  \item \code{eval.metric="auc"}: gives the average in-class area under the ROC curve.
#'  \item \code{eval.metric="logloss"}: gives the cross-entropy or logarithmic loss. 
#' }
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
#' @export
fastknnCV <- function(x, y, k = 3:15, method = "dist", folds = 5, 
                      eval.metric = "overall_acc") {
   
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
         y.hat <- fastknn(x[-te.idx,], y[-te.idx], x[te.idx,], k, method)
         classLoss(actual = y[te.idx], predicted = y.hat$class, 
                   prob = y.hat$prob, metric = eval.metric)
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

#### Split data into folds using stratified sampling
createCVFolds <- function(y, n) {
   folds <- integer(length(y))
   for (i in levels(y)) {
      folds[which(y == i)] <- sample(cut(1:sum(y == i), breaks = n, labels = FALSE))
   }
   
   return(folds)
}

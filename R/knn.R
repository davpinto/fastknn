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
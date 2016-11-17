#' Fast KNN Classifier
#'
#' Fast k-Nearest Neighbour classifier build upon ANN, a high efficient 
#' \code{C++} library for nearest neighbour searching.
#'
#' @param xtr matrix containing the training instances.
#' @param xte matrix containing the test instances.
#' @param ytr factor array with the training labels.
#' @param k number of neighbours considered.
#' @param method method used to infer the class membership probabilities of the 
#' test instances. Choose \code{"dist"} (default) to compute probabilites from 
#' the inverse of the nearest neighbour distances. This method works as
#' a shrinkage estimator and provides a better predictive performance in general.
#' Or you can choose \code{"vote"} to compute probabilities from the frequency 
#' of the nearest neighbour labels.
#'
#' @return \code{list} with predictions for the test set:
#' \itemize{
#'  \item \code{class}: factor array of predicted classes.
#'  \item \code{prob}: matrix with predicted probabilities.
#' }
#'
#' @export
#' 
fastknn <- function(xtr, xte, ytr, k, method = "dist") {
   
   #### Default number of neighbours
   if (missing(k)) {
      k <- min(10, nrow(xtr) - 1)
   }
   
   #### Find nearest neighbours
   knn.search <- RANN::nn2(data = xtr, query = xte, k = k, treetype = 'kd', 
                           searchtype = 'standard')
   
   #### Compute class membership probabilities
   label.mat <- matrix(ytr[knn.search$nn.idx], ncol = k)
   knn.prob <- switch(
      method,
      ## P_y = sum(1/d(nn_y))
      'dist' = {
         sapply(levels(ytr), function(cl, d, y) {
            rowSums(1/d * (y == cl)) / rowSums(1/d)
         }, d = pmax(knn.search$nn.dists, 1e-15), y = label.mat, 
         simplify=FALSE, USE.NAMES=TRUE)
      },
      ## P_y = count(nn_y)
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
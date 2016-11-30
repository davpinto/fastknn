#' Nearest Neighbors Features
#'
#' Do \strong{feature engineering} on the original dataset and extract new features,
#' generating a new dataset. Since KNN is a nonlinear learner, it makes a 
#' nonlinear mapping from the original dataset, making possible to achieve 
#' a great classification performance using a simple linear model on the new 
#' features, like GLM or LDA.
#' 
#' This \strong{feature engineering} procedure generates \code{k * c} new 
#' features using the distances between each observation and its \code{k} 
#' nearest neighbors inside each class, where \code{c} is the number of class 
#' labels. The procedure can be summarized as follows:
#' \enumerate{
#'    \item Generate the first feature as the distances from the nearest 
#'    neighbor in the first class.
#'    \item Generate the second feature as the sum of distances from the 2 
#'    nearest neighbors inside the first class.
#'    \item Generate the third feature as the sum of distances from the 3 
#'    nearest neighbors inside the first class.    
#'    \item And so on.
#' }
#' Repeat it for each class to generate the \code{k * c} new features. For the 
#' new training set, a n-fold CV approach is used to avoid overfitting.
#' 
#' This procedure is not so simple. But this method provides a easy interface 
#' to do it, and is very fast.
#'
#' @param xtr matrix containing the training instances.
#' @param xte matrix containing the test instances.
#' @param ytr factor array with the training labels.
#' @param k number of neighbors considered (default is 5). This choice is 
#' directly related to the number of new features. So, be careful with it. A 
#' large \code{k} may increase a lot the computing time for big datasets.
#' @param folds number of folds (default is 5) or an array with fold ids between 
#' 1 and \code{n} identifying what fold each observation is in. The smallest 
#' value allowable is \code{nfolds=3}.
#'
#' @return \code{list} with the new data:
#' \itemize{
#'  \item \code{new.tr}: \code{matrix} with the new training instances.
#'  \item \code{new.te}: \code{matrix} with the new test instances.
#' }
#'
#' @export
knnExtract <- function(xtr, ytr, xte, k = 5, folds = 5) {
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
      if (folds > 10) {
         warning("The number of folds is greater than 10. It may take too much time.")
      }
      folds <- createCVFolds(ytr, n = folds)
   }
   
   ### Transform fold ids to factor
   folds <- factor(paste('fold', folds, sep = '_'), 
                   levels = paste('fold', sort(unique(folds)), sep = '_'))
   
   #### Extract features from training set
   ## n-fold CV is used to avoid overfitting
   message("Building new training set...")
   tr.feat <- pbapply::pblapply(levels(ytr), function(y.label) {
      ## Iterate over data folds
      cv.feat <- lapply(levels(folds), function(fold.id) {
         te.idx <- which(fold.id == folds)
         tr.idx <- base::intersect(
            base::setdiff(1:nrow(xtr), te.idx),
            which(ytr == y.label)
         )
         dist.mat <- RANN::nn2(data = xtr[tr.idx, ], query = xtr[te.idx, ], 
                               k = k, treetype = 'kd', 
                               searchtype = 'standard')$nn.dists
         cbind(te.idx, matrixStats::rowCumsums(dist.mat))
      })
      tr.new <- do.call('rbind', cv.feat)
      tr.new <- tr.new[order(tr.new[, 1, drop = TRUE]),]
      return(tr.new[, -1])
   })
   tr.feat <- round(do.call('cbind', tr.feat), 6)
   colnames(tr.feat) <- paste0("knn", 1:ncol(tr.feat))
   
   #### Extract features from test set
   message("Building new test set...")
   te.feat <- pbapply::pblapply(levels(ytr), function(y.label) {
      idx <- which(ytr == y.label)
      dist.mat <- RANN::nn2(data = xtr[idx, ], query = xte, k = k, 
                            treetype = 'kd', searchtype = 'standard')$nn.dists
      matrixStats::rowCumsums(dist.mat)
   })
   te.feat <- round(do.call('cbind', te.feat), 6)
   colnames(te.feat) <- paste0("knn", 1:ncol(te.feat))
   
   #### Force to free memory
   rm(list = c("xtr", "ytr", "xte"))
   gc()
   
   return(list(
      new.tr = tr.feat,
      new.te = te.feat
   ))
}
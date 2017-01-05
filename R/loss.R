#' Classification Performance Measure
#'
#' Compute classification performance according to common evaluation metrics: 
#' classification error, AUC and log-loss.
#'
#' There are four evaluation metrics available sor far:
#' \itemize{
#'  \item \code{eval.metric="overal_error"}: default option. It gives the 
#'  overall misclassification rate. It do not require the \code{prob} parameter.
#'  \item \code{eval.metric="mean_error"}: gives the mean per class 
#'  misclassification rate. It do not require the \code{prob} parameter.
#'  \item \code{eval.metric="auc"}: gives the mean per class area under the ROC 
#'  curve. It requires the \code{prob} parameter.
#'  \item \code{eval.metric="logloss"}: gives the cross-entropy or logarithmic 
#'  loss. It requires the \code{prob} parameter.
#' }
#'
#' @param actual factor array with the true class labels.
#' @param predicted factor array with the predicted class labels.
#' @param prob matrix with predicted class membership probabilities. 
#' Rows are observations and columns are classes. It is required to calculate 
#' AUC and log-loss.
#' @param eval.metric evaluation metric to be used. It can be one of 
#' \code{c("overall_error", "mean_error", "auc", "logloss")}. The default 
#' option is \code{"overall_error"}.
#'
#' @return The classification performance measure.
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
#' classLoss(actual = y.te, predicted = knn.out$class, eval.metric = "overall_error")
#' classLoss(actual = y.te, predicted = knn.out$class, prob = knn.out$prob, eval.metric = "logloss")
#' }
classLoss <- function(actual, predicted, prob, eval.metric = "overall_error") {
   #### Check args
   stopifnot(is.factor(actual))
   if (!missing(predicted)) {
      stopifnot(is.factor(predicted))
      if(length(actual) != length(predicted))
         stop("actual and predicted must be the same lengths")
      if(!all.equal(levels(actual), levels(predicted)))
         stop("actual and predicted must be the same levels (class labels)")
   }
   if (!missing(prob)) {
      stopifnot(is.matrix(prob))
      stopifnot(ncol(prob) == nlevels(actual))
   }
   stopifnot(eval.metric %in% c("overall_error", "mean_error", "auc", "logloss"))
   
   #### Choose loss function
   loss <- switch(
      eval.metric,
      "overall_error" = {
         if (missing(predicted))
            stop("parameter 'predicted' is missing")
         1 - sum(actual == predicted) / length(actual)
      },
      "mean_error" = {
         if (missing(predicted))
            stop("parameter 'predicted' is missing")
         multiClassError(actual, predicted)
      },
      "auc" = {
         if (missing(prob))
            stop("parameter 'prob' is missing")
         multiAUC(actual, prob)
      },
      "logloss" = {
         if (missing(prob))
            stop("parameter 'prob' is missing")
         multiLogLoss(actual, prob)
      }
   )
   
   return(loss)
}

#### Multiclass error rate
multiClassError <- function(actual, predicted) {
   y <- decodeLabels(actual)
   yhat <- decodeLabels(predicted)
   mean(sapply(1:ncol(y), function(col.idx, y, yhat) {
      1 - sum(yhat[, col.idx] == y[, col.idx]) / nrow(y)
   }, y = y, yhat = yhat, simplify = TRUE))
}

#### Multiclass AUC
multiAUC <- function(actual, predicted) {
   y.mat <- decodeLabels(actual)
   mean(sapply(1:ncol(y.mat), function(col.idx, y, prob) {
      Metrics::auc(actual = y[, col.idx], predicted = prob[, col.idx])
   }, y = y.mat, prob = predicted, simplify = TRUE))
}

#### Multiclass Log-Loss
multiLogLoss <- function(actual, predicted, eps = 1e-15) {
   ## Avoid extreme (0 and 1) probabilities
   predicted <- pmin(pmax(predicted, eps), 1-eps);
   
   ## Decode class labels
   y.mat <- decodeLabels(actual)
   
   ## Compute log-loss
   log.loss <- (-1/nrow(y.mat)) * sum(y.mat*log(predicted))
   
   return(log.loss)
}

#### Transform labels into a binary matrix
decodeLabels <- function(y) {
   y.mat <-  as.data.frame.matrix(
      table(1:length(y), y)
   )
   
   return(y.mat)
}

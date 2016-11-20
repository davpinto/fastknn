#### Classification losses
classLoss <- function(actual, predicted, prob, metric = "overall_error") {
   loss <- switch(
      metric,
      "overall_error" = {
         1 - sum(actual == predicted) / length(actual)
      },
      "mean_error" = {
         multiClassError(actual, predicted)
      },
      "auc" = {
         multiAUC(actual, prob)
      },
      "logloss" = {
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
   y.mat <- 1:length(y) %>% 
      table(y) %>% 
      as.data.frame.matrix()
   
   return(y.mat)
}

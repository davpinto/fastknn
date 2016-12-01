#' Classification Decision Boundary
#'
#' Decision boundary of the \code{fastknn} classifier.
#'
#' @param xtr matrix containing the training instances. If \code{xtr} is not a 
#' bidimensional matrix only the first two columns will be considered.
#' @param ytr factor array with the training labels.
#' @param xte (optional) Matrix containing the test instances. The test points will be 
#' plotted over the surface boundary. If missing, the training points will be 
#' plotted instead. If \code{xte} is not a bidimensional matrix only the first 
#' two columns will be considered.
#' @param yte (optional) Factor array with the test labels.
#' @param k number of neighbors considered.
#' @param method method used to infer the class membership probabilities of the 
#' test instances. See \code{\link{fastknn}} for more details.
#' @param dpi a scalar that defines the graph resolution (default = 150). 
#' It means that \code{dpi^2} data points will be generated from the original 
#' dataset to draw the decision boundary. So, for large values (>= 300) it may 
#' take too much time to plot.
#'
#' @return \code{ggplot2} object.
#'
#' @author 
#' David Pinto.
#'
#' @export
#' 
#' @examples
#' \dontrun{
#' library("caTools")
#' library("fastknn")
#' 
#' data("spirals")
#' 
#' x <- data.matrix(spirals$x)
#' y <- spirals$y
#' 
#' set.seed(2048)
#' tr.idx <- which(sample.split(Y = y, SplitRatio = 0.7))
#' x.tr <- x[tr.idx,]
#' x.te <- x[-tr.idx,]
#' y.tr <- y[tr.idx]
#' y.te <- y[-tr.idx]
#' 
#' knnDecision(xtr = x.tr, ytr = y.tr, xte = x.te, yte = y.te, k = 10)
#' }
knnDecision <- function(xtr, ytr, xte, yte, k, method = "dist", dpi = 150) {
   
   #### Resample data
   x1 <- seq(
      from   = min(xtr[,1]) - 0.1*diff(range(xtr[,1])),
      to     = max(xtr[,1]) + 0.1*diff(range(xtr[,1])), 
      length = dpi
   )
   x2 <- seq(
      from   = min(xtr[,2]) - 0.1*diff(range(xtr[,2])),
      to     = max(xtr[,2]) + 0.1*diff(range(xtr[,2])), 
      length = dpi
   )
   
   #### Fit decision boundary
   x.new <- cbind(
      rep(x1, times = length(x2)), 
      rep(x2, each = length(x1))
   )
   if (missing(xte) | missing(yte)) {
      points.df <- data.frame(x1 = xtr[,1], x2 = xtr[,2], label = ytr)
   } else {
      points.df <- data.frame(x1 = xte[,1], x2 = xte[,2], label = yte)
   }
   y.hat <- fastknn(xtr[, 1:2, drop = FALSE], ytr, x.new, k, method)
   if (nlevels(ytr) > 2) {
      decision.df <- data.frame(x1 = x.new[,1], x2 = x.new[,2], 
                                y = y.hat$class, 
                                z = as.integer(y.hat$class) - 1)
   } else {
      decision.df <- data.frame(x1 = x.new[,1], x2 = x.new[,2], 
                                y = y.hat$prob[,1], 
                                z = as.integer(y.hat$prob[,1] >= 0.5))
   }
   
   #### Plot decision boundary
   g <- ggplot(data = decision.df) + 
      geom_tile(aes_string("x1", "x2", fill = "y"), color = 'transparent', 
                size = 0, alpha = 0.6) +
      viridis::scale_fill_viridis(guide = 'none', end = 0.6, option = "D",
                                  discrete = ifelse(nlevels(ytr) > 2, TRUE, FALSE)) +
      geom_contour(aes_string("x1", "x2", z = "z"), color = 'white', alpha = 0.6, 
                   size = 0.5, bins = nlevels(ytr) - 1) +
      geom_point(data = points.df, aes_string("x1", "x2", color = "label"), 
                 alpha = 1, size = 1) + 
      viridis::scale_color_viridis(guide = 'none', end = 0.6, option = "D",
                                   discrete = TRUE) +
      labs(x = expression(x[1]), y = expression(x[2]))
   
   return(g)
}

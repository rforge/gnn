### Plot loss ##################################################################

##' @title Plot of the Loss per Epoch after Training
##' @param x object of class "gnn_GNN"
##' @param plot.type character string indicating the type of plot
##' @param max.n.samples maximal number of samples to be plotted
##' @param type see ?plot
##' @param xlab see ?plot
##' @param ylab see ?plot
##' @param y2lab secondary y-axis label
##' @param labels see ?pairs
##' @param pair numeric(2) providing the indices of the pair being plotted (if
##'        provided)
##' @param ... additional arguments passed to the underlying plot() or pairs()
##' @return plot of the specified plot.type by side-effect
##' @author Marius Hofert
##' @note - plot() is already a generic, no need to define this generic for
##'         plot.gnn_GNN()
##'       - Could also add training time, but not directly loss related
##'       - For the naming of 'plot.type', see ?plot.ts
plot.gnn_GNN <- function(x, plot.type = c("scatter", "loss"), max.n.samples = NULL,
                         type = NULL, xlab = NULL, ylab = NULL, # for plot()
                         y2lab = NULL, # secondary y-axis label
                         labels = "X", pair = NULL, # for pairs()
                         ...)
{
    plot.type <- match.arg(plot.type)
    switch(plot.type, # currently only 'loss' available
           "scatter" = {
               ## Check
               if(!is.trained.gnn_GNN(x)) {
                   stop("'x' needs to be a trained object of class \"gnn_GNN\" if plot.type = \"scatter\"")
               }

               ## Evaluate GNN on the saved prior
               d <- tail(dim(x), n = 1) # output dimension
               pair.given <- !is.null(pair)
               if(is.null(max.n.samples))
                   max.n.samples <- if(d == 2 || pair.given) 5000 else 1000
               sample <- ffGNN(x, data = x[["prior"]][seq_len(min(nrow(x[["prior"]]), max.n.samples)),]) # at most max.n.samples samples

               ## Labels
               len.labs <- length(labels)
               stopifnot(len.labs == 1 || len.labs == d) # if d, then use 'labels' as is
               if(len.labs == 1) {
                   labels <- as.expression(lapply(1:d, function(j)
                       substitute(l.[j.], list(l. = as.name(labels[j]), j. = j))))
               }
               if(is.null(y2lab)) {
                   y2lab <- paste0("Sample size = ",nrow(sample))
               }

               ## If d = 2 or 'pair' is provided, use plot(), otherwise pairs()
               if(d == 2 || pair.given) {
                   if(pair.given) {
                       stopifnot(length(pair) == 2, 1 <= pair & pair <= d, pair[1] != pair[2])
                   } else {
                       pair <- c(1, 2)
                   }
                   ## Define variables
                   if(is.null(type)) type <- "p"
                   if(is.null(xlab)) xlab <- labels[pair[1]]
                   if(is.null(ylab)) ylab <- labels[pair[2]]
                   ## Plots
                   plot(sample[, pair], type = type, xlab = xlab, ylab = ylab, ...)
                   if(y2lab != "")
                       mtext(y2lab, side = 4, line = 0.5, adj = 0)
               } else if(d > 2) { # pairs plot
                   pairs(sample, gap = 0, labels = labels, ...)
                   mtext(y2lab, side = 4, line = 1.1, adj = 0.1)
               } else stop("Wrong 'd'")
           },
           "loss" = {
               ## Check
               if(!is.trained.gnn_GNN(x))
                   stop("'x' needs to be a trained object of class \"gnn_GNN\" if plot.type = \"loss\"")
                   ## Define variables
                   if(is.null(type)) type <- "l"
                   if(is.null(xlab)) xlab <- "Epoch"
                   if(is.null(ylab)) ylab <- paste0("Loss (",x[["loss.type"]],")")
                   if(is.null(y2lab)) {
                       y2lab <- paste0("Training set size = ",x[["n.train"]],
                                       ", dimension = ",x[["dim"]][1],
                                       ", batch size = ",x[["batch.size"]])
                   }
                   ## Plot
                   plot(x[["loss"]], type = type, xlab = xlab, ylab = ylab, ...)
                   if(y2lab != "") mtext(y2lab, side = 4, line = 0.5, adj = 0)
           },
           stop("Wrong 'plot.type'"))
}

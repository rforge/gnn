### Plot loss ##################################################################

##' @title Plot of the Loss per Epoch after Training
##' @param x object of class "gnn_GNN"
##' @param xlab x-axis label
##' @param ylab y-axis label
##' @param ... additional arguments passed to the underlying plot()
##' @return loss per epoch (invisibly)
##' @author Marius Hofert
##' @note - Could allow to plot (and return) various loss functions (if
##'         'x' is a list), but probably of little value
##'       - Could also add training time, but not directly loss related
plot_loss <- function(x, type = "l", xlab = "Epoch", ylab = "Loss", y2lab = NULL, ...)
{
    if(!is.trained.gnn_GNN(x))
        stop("'x' needs to be a trained object of class \"gnn_GNN\"")
    plot(x[["loss"]], type = type, xlab = xlab, ylab = ylab, ...)
    if(is.null(y2lab))
        y2lab <- paste0("Training set size = ",x[["n.train"]],
                        ", dimension = ",x[["dim"]][1],
                        ", batch size = ",x[["batch.size"]])
    mtext(y2lab, side = 4, line = 0.5, adj = 0)
    invisible(x[["loss"]])
}

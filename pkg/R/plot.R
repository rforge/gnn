### Plot loss ##################################################################

##' @title Plot of the Loss per Epoch after Training
##' @param x object of class "gnn_GNN"
##' @param plot.type character string indicating the type of plot
##' @param type see ?plot
##' @param xlab see ?plot
##' @param ylab see ?plot
##' @param y2lab secondary y-axis label
##' @param ... additional arguments passed to the underlying plot()
##' @return plot of the specified plot.type by side-effect; invisibly
##'         depending on 'plot.type'
##' @author Marius Hofert
##' @note - plot() is already a generic, no need to define it
##'       - Could also add training time, but not directly loss related
##'       - For 'plot.type', see ?plot.ts
plot.gnn_GNN <- function(x, plot.type = "loss", type = "l", xlab = "Epoch",
                         ylab = paste0("Loss (",x[["loss.type"]],")"),
                         y2lab = NULL, ...)
{
    plot.type <- match.arg(plot.type)
    switch(plot.type, # currently only 'loss' available
           "loss" = {
               if(!is.trained.gnn_GNN(x))
                   stop("'x' needs to be a trained object of class \"gnn_GNN\" if plot.type = \"loss\"")
               plot(x[["loss"]], type = type, xlab = xlab, ylab = ylab, ...)
               if(is.null(y2lab))
                   y2lab <- paste0("Training set size = ",x[["n.train"]],
                                   ", dimension = ",x[["dim"]][1],
                                   ", batch size = ",x[["batch.size"]])
               mtext(y2lab, side = 4, line = 0.5, adj = 0)
               invisible(x[["loss"]])
           },
           stop("Wrong 'plot.type'"))
}

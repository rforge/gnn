\name{GMMN_trained}
% Trained neural networks
\alias{VAE_gen_dim_2_300_560_ntrn_1965_nbat_100_nepo_300_freyfaces}
\alias(VAE_gen_dim_2_3_300_784_ntrn_60000_nbat_100_nepo_100_fmnist}

\description{
The generator component of trained variational autoencoders (VAEs). These VAE generators are used in the 
the vignette \code{VAE}.
}

\usage{
data("VAE_gen_dim_2_300_560_ntrn_1965_nbat_100_nepo_300_freyfaces")
data("VAE_gen_dim_2_3_300_784_ntrn_60000_nbat_100_nepo_100_fmnist")
}
\format{
\describe{
\item{VAE_gen_dim_2_300_560_ntrn_1965_nbat_100_nepo_300_freyfaces}{
\code{\link{raw}} \R object representing the generator component of a VAE (output
      layer is 560-dimensional, the single hidden layer is
      300-dimensional and the latent layer is two-dimensional) trained on 1965 frey faces (with batch size 100 and 300 epochs).
}
\item{VAE_gen_dim_2_3_300_784_ntrn_60000_nbat_100_nepo_100_fmnist}{
code{\link{raw}} \R object representing the generator component of a VAE (output
      layer is 784-dimensional, the three hidden layers are each
      300-dimensional and the latent layer is two-dimensional) trained on 60000 fashion mnist images (with batch size 100 and 100 epochs).
      }
     }
  }

\source{
  GPU server with NVIDIA Tesla P100 GPUs.
}
\author{Marius Hofert and Avinash Prasad}
\seealso{
  \code{\link{VAE_model}()}}
  
\keyword{datasets}

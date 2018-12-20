\name{GMMN_trained}
% Trained neural networks
\alias{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_C_0.25}
\alias{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_C_0.5}
\alias{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_C_0.75}
\alias{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_G_0.25}
\alias{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_G_0.5}
\alias{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_G_0.75}
\alias{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_mixCrt4}
\alias{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_mixGrt4}
\alias{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_mixMOrt4}
\alias{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_MO}
\alias{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_t4_0.25}
\alias{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_t4_0.5}
\alias{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_t4_0.75}
\alias{GMMN_dim_3_300_3_ntrn_60000_nbat_5000_nepo_300_NC21}
\alias{GMMN_dim_3_300_3_ntrn_60000_nbat_5000_nepo_300_NG21}
\alias{GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_C}
\alias{GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_G}
\alias{GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_NC23}
\alias{GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_NG23}
\alias{GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_t4}
\alias{GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_C}
\alias{GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_G}
\alias{GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_NC55}
\alias{GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_NG55}
\alias{GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_t4}
% Results obtained from trained NNs
\alias{GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_CvM_est}
\alias{GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_ES99_var_est}
\title{Trained Generative Moment Matching Networks}
\description{
  Trained generative moment matching networks (GMMNs) and
  related results. The results and some of the GMMNs
  are used in the vignette \code{GMMN_QRNG}.
}
\usage{
% Trained neural networks
data("GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_C_0.25")
data("GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_C_0.5")
data("GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_C_0.75")
data("GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_G_0.25")
data("GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_G_0.5")
data("GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_G_0.75")
data("GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_mixCrt4")
data("GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_mixGrt4")
data("GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_mixMOrt4")
data("GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_MO")
data("GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_t4_0.25")
data("GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_t4_0.5")
data("GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_t4_0.75")
data("GMMN_dim_3_300_3_ntrn_60000_nbat_5000_nepo_300_NC21")
data("GMMN_dim_3_300_3_ntrn_60000_nbat_5000_nepo_300_NG21")
data("GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_C")
data("GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_G")
data("GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_NC23")
data("GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_NG23")
data("GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_t4")
data("GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_C")
data("GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_G")
data("GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_NC55")
data("GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_NG55")
data("GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_t4")

% Results obtained from trained NNs
data("GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_CvM_est")
data("GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_ES99_var_est")
}
\format{
  % Trained neural networks
  \describe{
    \item{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_C_0.25}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are two-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a bivariate Clayton
      copula (with parameter chosen such that Kendall's tau equals 0.25).
    }
    \item{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_C_0.5}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are two-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a bivariate Clayton
      copula (with parameter chosen such that Kendall's tau equals 0.5).
    }
    \item{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_C_0.75}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are two-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a bivariate Clayton
      copula (with parameter chosen such that Kendall's tau equals 0.75).
    }
    \item{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_G_0.25}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are two-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a bivariate Gumbel
      copula (with parameter chosen such that Kendall's tau equals 0.25).
    }
    \item{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_G_0.5}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are two-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a bivariate Gumbel
      copula (with parameter chosen such that Kendall's tau equals 0.5).
    }
    \item{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_G_0.75}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are two-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a bivariate Gumbel
      copula (with parameter chosen such that Kendall's tau equals 0.75).
    }
    \item{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_mixCrt4}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are two-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a bivariate half-half mixture of a Clayton
      copula (with parameter chosen such that Kendall's tau equals 0.5)
      and a rotated (by 90 degree) $t$ copula (with 4 degrees of freedom
      and correlation parameter chosen such that Kendall's tau equals
      0.5); see \code{vignette("GMMN_QRNG", package = "gnn")}.
    }
    \item{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_mixGrt4}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are two-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a bivariate half-half mixture of a Gumbel
      copula (with parameter chosen such that Kendall's tau equals 0.5)
      and a rotated (by 90 degree) $t$ copula (with 4 degrees of freedom
      and correlation parameter chosen such that Kendall's tau equals
      0.5).
    }
    \item{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_mixMOrt4}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are two-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a bivariate half-half mixture of a Marshall--Olkin
      copula (with \eqn{\alpha_{1}=0.75}{alpha_1 = 0.75} and
      \eqn{\alpha_{2}=0.60}{alpha_2 = 0.60})
      and a rotated (by 90 degree) $t$ copula (with 4 degrees of freedom
      and correlation parameter chosen such that Kendall's tau equals
      0.5).
    }
    \item{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_MO}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are two-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a Marshall--Olkin
      copula (with \eqn{\alpha_{1}=0.75}{alpha_1=0.75} and
      \eqn{\alpha_{2}=0.60}{alpha_2=0.60}).
    }
    \item{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_t4_0.25}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are two-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a two-dimensional $t$ copula (with 4 degrees of freedom
      and equi-correlation parameter chosen such that Kendall's tau equals
      0.25).
    }
    \item{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_t4_0.5}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are two-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a two-dimensional $t$ copula (with 4 degrees of freedom
      and equi-correlation parameter chosen such that Kendall's tau equals
      0.5).
    }
    \item{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_t4_0.75}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are two-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a two-dimensional $t$ copula (with 4 degrees of freedom
      and equi-correlation parameter chosen such that Kendall's tau equals
      0.75).
    }
    \item{GMMN_dim_3_300_3_ntrn_60000_nbat_5000_nepo_300_NC21}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are three-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a three-dimensional nested Clayton copula
      (with sector dimensions 2 and 1, corresponding Kendall's tau 0.5
      within the first sector and Kendall's tau 0.25 between the two sectors).
    }
    \item{GMMN_dim_3_300_3_ntrn_60000_nbat_5000_nepo_300_NG21}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are three-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a three-dimensional nested Gumbel copula
      (with sector dimensions 2 and 1, corresponding Kendall's tau 0.5
      within the first sector and Kendall's tau 0.25 between the two sectors).
    }
    \item{GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_C}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are five-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a five-dimensional Clayton
      copula (with parameter chosen such that Kendall's tau equals 0.5).
    }
    \item{GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_G}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are five-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a five-dimensional Gumbel
      copula (with parameter chosen such that Kendall's tau equals 0.5).
    }
    \item{GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_NC23}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are five-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a five-dimensional nested Clayton copula
      (with sector dimensions 2 and 3, corresponding Kendall's tau 0.5
      and 0.75, and Kendall's tau 0.25 between the two sectors).
    }
    \item{GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_NG23}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are five-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a five-dimensional nested Gumbel copula
      (with sector dimensions 2 and 3, corresponding Kendall's tau 0.5
      and 0.75, and Kendall's tau 0.25 between the two sectors);
      see \code{vignette("GMMN_QRNG", package = "gnn")}.
    }
    \item{GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_t4}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are five-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a five-dimensional $t$ copula (with 4 degrees of freedom
      and equi-correlation parameter chosen such that Kendall's tau equals
      0.5); see \code{vignette("GMMN_QRNG", package = "gnn")}.
    }
    \item{GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_C}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are 10-dimensional, the single hiddenlayer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a 10-dimensional Clayton
      copula (with parameter chosen such that Kendall's tau equals 0.5).
    }
    \item{GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_G}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are 10-dimensional, the single hiddenlayer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a 10-dimensional Gumbel
      copula (with parameter chosen such that Kendall's tau equals 0.5).
    }
    \item{GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_NC55}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are 10-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a 10-dimensional nested Clayton copula
      (with sector dimensions 5 and 5, corresponding Kendall's tau 0.5
      and 0.75, and Kendall's tau 0.25 between the two sectors).
    }
    \item{GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_NG55}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are 10-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a 10-dimensional nested Gumbel copula
      (with sector dimensions 5 and 5, corresponding Kendall's tau 0.5
      and 0.75, and Kendall's tau 0.25 between the two sectors).
    }
    \item{GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_t4}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are 10-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a 10-dimensional $t$ copula (with 4 degrees of freedom
      and equi-correlation parameter chosen such that Kendall's tau equals
      0.5).
    }

    % Results obtained from trained NNs
    \item{GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_CvM_est}{
      (4, 100)-\code{\link{matrix}} containing, in each row, 100 replications of
      computed Cramer-von Mises statistics based on 1000 pseudo-samples
      generated from a pseudo-random number generator (first and third
      row) and the trained GMMN (second and fourth row) of the
      five-dimensional $t$ copula (first two rows) and the nested
      Gumbel copula (last two rows) as described above;
      see \code{vignette("GMMN_QRNG", package = "gnn")}.
    }
    \item{GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_ES99_var_est}{
      (5, 19)-\code{\link{matrix}} containing, in each row, the
      variance estimates of expected shortfall at 99\%
      confidence level of the sum of five-dimensional distributions
      (from 25 replications, for the sample sizes 512, 724, 1024, 1448,
      2048, 2896, 4096, 5793, 8192, 11585, 16384, 23170, 32768, 46341,
      65536, 92682, 131072, 185364, 262144 which are
      roughly equidistant in log-scale) based on pseudo-samples (so
      based on the classical Monte Carlo estimator; first and
      fourth row) and quasi-samples (obtained from the GMMN
      in the second and fifth row and obtained from a
      quasi-random number generator in the third row). The
      five-dimensional distributions have standard
      normal margins and $t$ copula (first three rows)
      and nested Gumbel copula (last two rows) as described above;
      see \code{vignette("GMMN_QRNG", package = "gnn")}.
    }
  }
}
\source{
  GPU server with NVIDIA Tesla P100 GPUs.
}
\author{Marius Hofert and Avinash Prasad}
\references{
  Hofert, M., Prasad, A. and Zhu, M. (2018).
  Quasi-random number generators for multivariate distributions
  based on generative neural networks. See https://arxiv.org/abs/1811.00683
%% \emph{myjournal} \bold{myversion}(mynumber), --.
}
\seealso{
  \code{\link{GMMN_model}()}
}
\examples{
ngen <- 1000L # sample size of the generated date

## Generate a pseudo-sample from the mixture copula considered here
library(copula)
C.cop <- claytonCopula(iTau(claytonCopula(), tau = 0.5)) # Clayton copula
t.cop <- tCopula(iTau(tCopula(), tau = 0.5), df = 4) # t copula
t90.cop <- rotCopula(t.cop, flip = c(TRUE, FALSE)) # t copula rotated by 90 degrees
mix.cop <- mixCopula(list(C.cop, t90.cop), w = c(1/2, 1/2)) # the mixture copula
set.seed(271)
U.mix.PRNG <- rCopula(ngen, copula = mix.cop)

## Get a trained GMMN
NN <- read_dataset("GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_mixCrt4")
library(keras)
GMMN.mix <- unserialize_model(NN, custom_objects = c(loss = loss))

## Generate an approximate pseudo- and quasi-sample from GMMN.mix
## First, sample from the prior distribution (pseudo-random and quasi-random numbers)
N01.prior.PRNG <- matrix(rnorm(ngen * 2), ncol = 2) # N(0,1) PRNG
library(qrng)
N01.prior.QRNG <- qnorm(sobol(ngen, d = 2, randomize = TRUE)) # N(0,1) QRNG
## Generate data from the fitted GMMN
U.mix.GMMN.PRNG <- pobs(predict(GMMN.mix, x = N01.prior.PRNG))
U.mix.GMMN.QRNG <- pobs(predict(GMMN.mix, x = N01.prior.QRNG))
## Note: for pobs(), see Hofert, Prasad and Zhu (2018)

## Pseudo-random sample from the mixture copula, pseudo-random sample
## from the trained GMMN and quasi-random sample from the trained GMMN
layout(t(1:3)) # 1 x 3 layout
opar <- par(pty = "s") # square plots
plot(U.mix.PRNG,      main = "PRNG sample",      xlab = bquote(U[1]), ylab = bquote(U[2]))
plot(U.mix.GMMN.PRNG, main = "GMMN PRNG sample", xlab = bquote(U[1]), ylab = bquote(U[2]))
plot(U.mix.GMMN.QRNG, main = "GMMN QRNG sample", xlab = bquote(U[1]), ylab = bquote(U[2]))
par(opar) # restore graphical parameters
layout(1) # restore layout
}
\keyword{datasets}

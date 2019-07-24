\name{GMMN_trained}
% Trained neural networks
\alias{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_C_tau_0.25}
\alias{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_C_tau_0.5}
\alias{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_C_tau_0.75}
\alias{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_G_tau_0.25}
\alias{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_G_tau_0.5}
\alias{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_G_tau_0.75}
\alias{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_eqmix_C_tau_0.5_rot90_t4_tau_0.5}
\alias{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_eqmix_G_tau_0.5_rot90_t4_tau_0.5}
\alias{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_eqmix_MO_0.75_0.6_rot90_t4_tau_0.5}
\alias{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_MO_0.75_0.6}
\alias{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_t4_tau_0.25}
\alias{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_t4_tau_0.5}
\alias{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_t4_tau_0.75}
\alias{GMMN_dim_3_300_3_ntrn_60000_nbat_5000_nepo_300_NC21_tau_0.25_0.5_0.75}
\alias{GMMN_dim_3_300_3_ntrn_60000_nbat_5000_nepo_300_NG21_tau_0.25_0.5_0.75}
\alias{GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_C_tau_0.5}
\alias{GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_G_tau_0.5}
\alias{GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_NC23_tau_0.25_0.5_0.75}
\alias{GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_NG23_tau_0.25_0.5_0.75}
\alias{GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_t4_tau_0.5}
\alias{GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_C_tau_0.5}
\alias{GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_G_tau_0.5}
\alias{GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_NC55_tau_0.25_0.5_0.75}
\alias{GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_NG55_tau_0.25_0.5_0.75}
\alias{GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_t4_tau_0.5}
% Results obtained from trained NNs
\alias{GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_CvM_statistics_t4_tau_0.5_G_tau_0.5}
\alias{GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_ES99_var_estimates}
\title{Trained Generative Moment Matching Networks}
\description{
  Trained generative moment matching networks (GMMNs) and
  related results. The results and some of the GMMNs
  are used in the vignette \code{GMMN_QRNG}.
}
\usage{
% Trained neural networks
data("GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_C_tau_0.25")
data("GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_C_tau_0.5")
data("GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_C_tau_0.75")
data("GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_G_tau_0.25")
data("GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_G_tau_0.5")
data("GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_G_tau_0.75")
data("GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_eqmix_C_tau_0.5_rot90_t4_tau_0.5")
data("GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_eqmix_G_tau_0.5_rot90_t4_tau_0.5")
data("GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_eqmix_MO_0.75_0.6_rot90_t4_tau_0.5")
data("GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_MO_0.75_0.6")
data("GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_t4_tau_0.25")
data("GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_t4_tau_0.5")
data("GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_t4_tau_0.75")
data("GMMN_dim_3_300_3_ntrn_60000_nbat_5000_nepo_300_NC21_tau_0.25_0.5_0.75")
data("GMMN_dim_3_300_3_ntrn_60000_nbat_5000_nepo_300_NG21_tau_0.25_0.5_0.75")
data("GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_C_tau_0.5")
data("GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_G_tau_0.5")
data("GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_NC23_tau_0.25_0.5_0.75")
data("GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_NG23_tau_0.25_0.5_0.75")
data("GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_t4_tau_0.5")
data("GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_C_tau_0.5")
data("GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_G_tau_0.5")
data("GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_NC55_tau_0.25_0.5_0.75")
data("GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_NG55_tau_0.25_0.5_0.75")
data("GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_t4_tau_0.5")

% Results obtained from trained NNs
data("GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_CvM_statistics_t4_tau_0.5_G_tau_0.5")
data("GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_ES99_var_estimates")
}
\format{
  % Trained neural networks
  \describe{
    \item{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_C_tau_0.25}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are two-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a bivariate Clayton
      copula (with parameter chosen such that Kendall's tau equals 0.25).
    }
    \item{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_C_tau_0.5}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are two-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a bivariate Clayton
      copula (with parameter chosen such that Kendall's tau equals 0.5).
    }
    \item{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_C_tau_0.75}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are two-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a bivariate Clayton
      copula (with parameter chosen such that Kendall's tau equals 0.75).
    }
    \item{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_G_tau_0.25}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are two-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a bivariate Gumbel
      copula (with parameter chosen such that Kendall's tau equals 0.25).
    }
    \item{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_G_tau_0.5}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are two-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a bivariate Gumbel
      copula (with parameter chosen such that Kendall's tau equals 0.5).
    }
    \item{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_G_tau_0.75}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are two-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a bivariate Gumbel
      copula (with parameter chosen such that Kendall's tau equals 0.75).
    }
    \item{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_eqmix_C_tau_0.5_rot90_t4_tau_0.5}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are two-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a bivariate half-half mixture of a Clayton
      copula (with parameter chosen such that Kendall's tau equals 0.5)
      and a rotated (by 90 degree) $t$ copula (with 4 degrees of freedom
      and correlation parameter chosen such that Kendall's tau equals
      0.5); see \code{vignette("GMMN_QRNG", package = "gnn")}.
    }
    \item{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_eqmix_G_tau_0.5_rot90_t4_tau_0.5}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are two-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a bivariate half-half mixture of a Gumbel
      copula (with parameter chosen such that Kendall's tau equals 0.5)
      and a rotated (by 90 degree) $t$ copula (with 4 degrees of freedom
      and correlation parameter chosen such that Kendall's tau equals
      0.5).
    }
    \item{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_eqmix_MO_0.75_0.6_rot90_t4_tau_0.5}{
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
    \item{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_MO_0.75_0.6}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are two-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a Marshall--Olkin
      copula (with \eqn{\alpha_{1}=0.75}{alpha_1=0.75} and
      \eqn{\alpha_{2}=0.60}{alpha_2=0.60}).
    }
    \item{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_t4_tau_0.25}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are two-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a two-dimensional $t$ copula (with 4 degrees of freedom
      and equi-correlation parameter chosen such that Kendall's tau equals
      0.25).
    }
    \item{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_t4_tau_0.5}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are two-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a two-dimensional $t$ copula (with 4 degrees of freedom
      and equi-correlation parameter chosen such that Kendall's tau equals
      0.5).
    }
    \item{GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_t4_tau_0.75}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are two-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a two-dimensional $t$ copula (with 4 degrees of freedom
      and equi-correlation parameter chosen such that Kendall's tau equals
      0.75).
    }
    \item{GMMN_dim_3_300_3_ntrn_60000_nbat_5000_nepo_300_NC21_tau_0.25_0.5_0.75}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are three-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a three-dimensional nested Clayton copula
      (with sector dimensions 2 and 1, corresponding Kendall's tau 0.5
      within the first sector and Kendall's tau 0.25 between the two sectors).
    }
    \item{GMMN_dim_3_300_3_ntrn_60000_nbat_5000_nepo_300_NG21_tau_0.25_0.5_0.75}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are three-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a three-dimensional nested Gumbel copula
      (with sector dimensions 2 and 1, corresponding Kendall's tau 0.5
      within the first sector and Kendall's tau 0.25 between the two sectors).
    }
    \item{GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_C_tau_0.5}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are five-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a five-dimensional Clayton
      copula (with parameter chosen such that Kendall's tau equals 0.5).
    }
    \item{GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_G_tau_0.5}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are five-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a five-dimensional Gumbel
      copula (with parameter chosen such that Kendall's tau equals 0.5).
    }
    \item{GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_NC23_tau_0.25_0.5_0.75}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are five-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a five-dimensional nested Clayton copula
      (with sector dimensions 2 and 3, corresponding Kendall's tau 0.5
      and 0.75, and Kendall's tau 0.25 between the two sectors).
    }
    \item{GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_NG23_tau_0.25_0.5_0.75}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are five-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a five-dimensional nested Gumbel copula
      (with sector dimensions 2 and 3, corresponding Kendall's tau 0.5
      and 0.75, and Kendall's tau 0.25 between the two sectors);
      see \code{vignette("GMMN_QRNG", package = "gnn")}.
    }
    \item{GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_t4_tau_0.5}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are five-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a five-dimensional $t$ copula (with 4 degrees of freedom
      and equi-correlation parameter chosen such that Kendall's tau equals
      0.5); see \code{vignette("GMMN_QRNG", package = "gnn")}.
    }
    \item{GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_C_tau_0.5}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are 10-dimensional, the single hiddenlayer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a 10-dimensional Clayton
      copula (with parameter chosen such that Kendall's tau equals 0.5).
    }
    \item{GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_G_tau_0.5}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are 10-dimensional, the single hiddenlayer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a 10-dimensional Gumbel
      copula (with parameter chosen such that Kendall's tau equals 0.5).
    }
    \item{GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_NC55_tau_0.25_0.5_0.75}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are 10-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a 10-dimensional nested Clayton copula
      (with sector dimensions 5 and 5, corresponding Kendall's tau 0.5
      and 0.75, and Kendall's tau 0.25 between the two sectors).
    }
    \item{GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_NG55_tau_0.25_0.5_0.75}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are 10-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a 10-dimensional nested Gumbel copula
      (with sector dimensions 5 and 5, corresponding Kendall's tau 0.5
      and 0.75, and Kendall's tau 0.25 between the two sectors).
    }
    \item{GMMN_dim_10_300_10_ntrn_60000_nbat_5000_nepo_300_t4_tau_0.5}{
      \code{\link{raw}} \R object representing a GMMN (input and output
      layer are 10-dimensional, the single hidden layer is
      300-dimensional) trained on 60000 pseudo-samples (with batch size 5000 and
      300 epochs) from a 10-dimensional $t$ copula (with 4 degrees of freedom
      and equi-correlation parameter chosen such that Kendall's tau equals
      0.5).
    }

    % Results obtained from trained NNs
    \item{GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_CvM_statistics_t4_tau_0.5_G_tau_0.5}{
      (100, 4)-\code{\link{matrix}} containing, in each column, 100 replications of
      computed Cramer-von Mises statistics based on 1000 pseudo-samples
      generated from a pseudo-random number generator (first and third
      column) and the trained GMMN (second and fourth column) of the
      five-dimensional $t$ copula (first two columns) and the nested
      Gumbel copula (last two columns) as described above;
      see \code{vignette("GMMN_QRNG", package = "gnn")}.
    }
    \item{GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_ES99_var_estimates}{
      (19, 5)-\code{\link{matrix}} containing, in each column, the
      variance estimates of expected shortfall at 99\%
      confidence level of the sum of the components of a
      five-dimensional distribution (standard normal margins;
      $t$ copula in the first three columns, Gumbel in the last two).
      The variance estimates are computed from 25 replications
      for the 19 sample sizes 512, 724, 1024,
      1448, 2048, 2896, 4096, 5793, 8192, 11585, 16384, 23170, 32768, 46341,
      65536, 92682, 131072, 185364, 262144 (roughly equidistant in
      log-scale); for each sample, the pseudo-observations are considered.
      First and fourth column contain variance estimates of the classical
      Monte Carlo estimator as they are based on a PRNG, second and
      fifth column contain the variance estimates for the GMMN-PRNG in that
      sense, and the third column contains the results for the GMMN-QRNG.
      See also \code{vignette("GMMN_QRNG", package = "gnn")}.
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
objname = "GMMN_dim_2_300_2_ntrn_60000_nbat_5000_nepo_300_eqmix_C_tau_0.5_rot90_t4_tau_0.5"
NN <- read_rda(objname, file = objname, package = "gnn")
library(keras)
GMMN.mix <- unserialize_model(NN, custom_objects = c(loss = loss))

## Generate an approximate pseudo- and quasi-sample from GMMN.mix
## First, sample from the prior distribution (pseudo-random and quasi-random numbers)
N01.prior.PRNG <- matrix(rnorm(ngen * 2), ncol = 2) # N(0,1) PRNG
library(qrng)
N01.prior.QRNG <- qnorm(sobol(ngen, d = 2, randomize = "digital.shift")) # N(0,1) QRNG
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

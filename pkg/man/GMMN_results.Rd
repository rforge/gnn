\name{GMMN_results}
% Results obtained from trained NNs
\alias{GMMN_QMC_res_CvMstats_GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_t4_tau_0.5}
\alias{GMMN_QMC_res_ES99_sd_GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_t4_tau_0.5}
\title{Results Obtained from Trained Generative Moment Matching Networks}
\description{
  Cached results from vignette \code{GMMN_QMC}.
}
\usage{
% Results obtained from trained NNs
data("GMMN_QMC_res_CvMstats_GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_t4_tau_0.5")
data("GMMN_QMC_res_ES99_sd_GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_t4_tau_0.5")
}
\format{
  \describe{
    % Results obtained from trained NNs
    \item{GMMN_QMC_res_CvMstats_GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_t4_tau_0.5}{
      (100, 3)-\code{\link{matrix}} containing, in each column, 100 replications of
      computed Cramer-von Mises statistics based on 1000 pseudo-samples
      generated from a pseudo-random number generator (first
      column), generated from a trained GMMN based on a pseudo-random
      number generator (second column) and generated from a trained GMMN
      based on a quasi-random number generator (third column) for a
      five-dimensional $t$ copula (with 4 degrees of freedom and
      equi-correlation parameter chosen such that Kendall's tau equals 0.5);
      see \code{vignette("GMMN_QMC", package = "gnn")}.
    }
    \item{GMMN_QMC_res_ES99_sd_GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_t4_tau_0.5}{
      (19, 4)-\code{\link{matrix}} containing, in each column, the
      standard deviation estimates of expected shortfall at 99\%
      confidence level of the sum of the components of a
      five-dimensional distribution with standard normal margins and
      $t_4$ copula.
      The standard deviation estimates are computed from 25 replications
      for the 19 sample sizes as indicated by the row names (they are
      roughly equidistant in log-scale). The first and fourth columns
      contain standard deviation
      estimates based on a copula PRNG and QRNG, respectively, whereas
      the third and forth column are based on a GMMN PRNG and GMMN QRNG,
      respectively. See also \code{vignette("GMMN_QMC", package = "gnn")}.
    }
  }
}
\source{
  Pre-trained NNs were used locally to obtain the results.
}
\author{Marius Hofert and Avinash Prasad}
\references{
  Hofert, M., Prasad, A. and Zhu, M. (2018).
  Quasi-Monte Carlo for multivariate distributions via generative neural networks.
  (See https://arxiv.org/abs/1811.00683 for an early version)
%% \emph{myjournal} \bold{myversion}(mynumber), --.
}
\seealso{
  \code{\link{GMMN_model}()}, \code{\link{train_once}()}
}
\examples{
data("GMMN_QMC_res_CvMstats_GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_t4_tau_0.5")
str(GMMN_QMC_res_CvMstats_GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_t4_tau_0.5)

data("GMMN_QMC_res_ES99_sd_GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_t4_tau_0.5")
str(GMMN_QMC_res_ES99_sd_GMMN_dim_5_300_5_ntrn_60000_nbat_5000_nepo_300_t4_tau_0.5)
}
\keyword{datasets}

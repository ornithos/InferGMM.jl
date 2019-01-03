# InferGMM.jl
Early stage of a project using Gaussian Mixture Models (GMMs) as the basis for various inference routines.

This package includes a custom implementation of GMM, ignoring the implementation in `Distributions.jl`. This is allowing me to experiment more freely with the implementation, and has been required for some substantial speed-ups (for instance saving on Cholesky decompositions on every constructor, faster sampling and faster llh calcs). The stable parts of this repo at the time of writing this note:

* gmm = GMM(mus::Matrix{T}, sigmas::Array{T,3}, pis::Vector{T}`.
* `rand(gmm, n)` (generate random variates from a GMM object).
* `logpdf(gmm, X)` (calculate the log density of `X` wrt a GMM object).
* `gmm_fit(X::Matrix{T}, gmm::GMM)` (Expectation-Maximisation fit of `gmm` to `X`. This implementation supports weighted samples (see `src/gmm.jl`).

There's a few other bits of functionality, but this repo is a WIP at present, and I will not make a serious effort to document it until I'm further down the track.

Alex

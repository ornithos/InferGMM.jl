# InferGMM.jl
Early stage of a project using Gaussian Mixture Models (GMMs) as the basis for various inference routines.

This package includes a custom implementation of GMM, ignoring the implementation in `Distributions.jl`. This has allowed me to experiment more freely with the implementation, and has been required for some substantial speed-ups (for instance saving on Cholesky decompositions on every constructor, faster sampling and faster llh calcs). The stable parts of this repo at the time of writing this note:

* gmm = GMM(mus::Matrix{T}, sigmas::Array{T,3}, pis::Vector{T}).
* `rand(gmm, n)` (generate random variates from a GMM object).
* `logpdf(gmm, X)` (calculate the log density of `X` wrt a GMM object).
* `gmm_fit(X::Matrix{T}, gmm::GMM)` (Expectation-Maximisation fit of `gmm` to `X`. This implementation supports weighted samples (see `src/gmm.jl`).

This repository is substantially a work-in-progress, and may be worth forking for stability if it is useful at present. I have found the idea of using a GMM for a proposal distribution / variational posterior to be an appealing one, but I have yet to find a circumstance in which it is fast enough to be especially worthwhile. I have some remaining ideas, which if I have time, should appear in this repo. Serious documentation is not likely to be forthcoming until this whole thing seems worthwhile.

# Inference Algorithms

We will use the ubiquitous banana function (in 2D) to demonstrate the algorithms:
```julia
function p_log_banana(X)
    x1 = X[1,:]./10.
    return -x1.^2/(2*0.03) - 0.5*(X[2,:] - 100.*(x1.^2 .- 0.03)).^2
end
```
In what follows, $q(x; \phi) = \sum_{j=1}^K \pi_j \mathcal{N}(\mu_j, \Sigma_j)$, sometimes written as $q_\phi$ where $x$ is obvious from context. For the time being, $\pi_j = \frac{1}{K}$ for all $j$, since it avoids some undesirable behaviour, is cleaner to write, and is anyway used for a proposal, for which component weights can easily be updated. An initial $q_\phi$ should be specified:
```julia
using Distributions, Random
using AxUtil, InferGMM
k = 5
gmm_init = GMM(randn(k,2), cat([AxUtil.Random.psd_matrix(2) for i in 1:k]...,dims=3)*0.2, rand(Dirichlet(ones(k)*10)));
```

Note that (Flux)[https://github.com/FluxML/Flux.jl] is used internally for automatic differentiation.

* **Variational Inference**, that is minimizing $\text{KL}(q_\phi \| p)$. The syntax for optimising $q_\phi = $ `gmm_init` is `optimise_components_vi(d::GMM, log_f::Function, epochs::Int, opt::Flux.ADAM, batch_size_per_cls::Int)`. Note that the sampling is stratified by the number of components, and hence one specifies the number of samples to evaluate the integral (`batch_size_per_cls`) *per component*. Example:

```julia
_opt = ADAM(8e-3)
gmm_banana, obj_history = InferGMM.optimise_components_vi(gmm_init, x->0.5p_log_banana(x), 1000, _opt, 3)
```

Notice that the target density has been annealed to the power $1/2$ to try to correct for VI's well known bias towards an 'inner approximation'.

* **Monte Carlo Objectives**, that is minimizing $\E_q(\frac{1}{M} \left[\sum{m=1}^M\log \frac{p(x^m)}{q(x^m)} \right]$. This is the sometimes known as the IWAE objective, as it was first introduced in Burda et al. (2016) for training variational autoencoders. However, there is no reason to restrict it to this application, and the language reflects that chosen by Mnih & Rezende (2016). It is fairly easy to adapt the VI procedure for this since the objectives are the same when $M=1$, but for legacy reasons, specifying `nproposals = 1` will fail. The syntax is: `optimise_components_mco(d::GMM, log_f::Function, epochs::Int, opt::Flux.ADAM, batch_size_per_cls::Int, nproposal_per_mco_smp::Int)`, where `nproposal_per_mco_smp` is $M$ in the previous equation. Some extra variance is introduced by sampling a single point $x$ from the `nproposals` rather than passing through all (weighted) points through the AD engine. Often this leads to substantial time savings, and Burda et. al (2016) discuss something similar. Example:

```julia
_opt = ADAM(8e-3)
gmm_banana, obj_history = InferGMM.optimise_components_vi(gmm_init, p_log_banana, 1000, _opt, 3, 5)
```

By using `batch_size_per_cls > 1`, one obtains one of the strategies used in Rainforth et al. 2018 ("Tighter Variational Bounds are not Necessarily Better"), or something close to it. If one wishes to use the 'Combination' strategy also advocated in this paper, this can be approximated by using the generic Bayes By Backprop (BBB) function and specifying the `bbb_opts` accordingly. For reference, these options are displayed in full at the bottom of this doc.

```julia
gmm2_, obj2_ = InferGMM.optimise_components_bbb(gmm_init, x->0.5p_log_banana(x), 1000,
    opts=InferGMM.variational.bbb_opts(opt=_opt, vi_batch_size_per_cls=3, mco_batch_size_per_cls=3,
        nproposal_per_mco_smp=5))
```
This would implicitly correspond to a convex combination where $\beta = 0.5 = \frac{3}{6}$, although note that it is different from the 'Combination' strategy since the latter uses the *same* particles for both the "VAE" (unweighted) and "IWAE" (weighted), whereas we use different ones *and* introduce sampling variance by resampling from the proposals. It's not obvious (to me) which is better, but I imagine some variance is reduced at least by using different particles between these strategies (otherwise one is performing something like 'Laplace smoothing' on the importance weights).

* **Reverse KL / Maximum Likelihood / 'Expectation Propagation'**, that is minimizing $\text{KL}(p \| q_\phi)$. This is done locally, so while the integral is in theory over $p$, we draw from $q_\phi$ and use importance weights. The variance introduced from this does not seem especially problematic, since when the Expected Sample Size (ESS) is $\approx 1$, we can expect it to be in a direction towards a high probability region. The only downside to this procedure is it doesn't use gradient information of the target, rather just adapting the GMMs towards the importance weighted observations, a little like a soft/gradient version of AMIS (Adaptive Mixture Importance Sampling). The signature of this function does not yet use the BBB options, as I find it less useful than some of the other inference algos:
`optimise_components_bbb_revkl(d::GMM, log_f::Function, epochs::Int, batch_size_per_cls::Int; opt::Flux.ADAM=ADAM(1e-3),ixs=nothing)`.


### All BBB Options:
This is a more general way to access all Lower Bound based inference procedures. An estimate of the lower bound is calculated from the samples generated, and a heuristic is used to determine if the objective is hit convergence. (Based on moving averages.)

```julia
opt::Flux.ADAM=ADAM(1e-3)
vi_batch_size_per_cls::Int = 4
mco_batch_size_per_cls::Int = 4
nproposal_per_mco_smp::Int = 3
converge_thrsh::AbstractFloat=0.999
auto_lr::Bool=true
anneal_sched::AbstractArray=[1.]
fix_mean::Bool=false
fix_cov::Bool=false
log_f_prev::Union{Function,Nothing}=nothing
```
We have discussed `vi_batch_size_per_cls`, `mco_batch_size_per_cls`, `nproposal_per_mco_smp`. The `converge_thrsh` is the amount the moving average must be above of a previous window of the moving average for convergence to be detected. `auto_lr` automatically reduces the learning rate if a divergence in objective value is observed. `anneal_sched` can be used in conjunction with `log_f_prev` in order to anneal between this and the target. The optimisation is targeting the distribution $\beta * \text{log_f} + (1-\beta) \text{log_f_prev}$ where $\beta$ is determined by the annealing schedule. `fix_mean`, `fix_cov` hold either the means or the covariances constant during optimisation (respectively).

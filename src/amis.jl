using Parameters, Formatting
using AxUtil
using InferGMM: GMM, gmm_fit, importance_sample, update, rmcomponents
using AxUtil.MCDiagnostic: is_eff_ss
using AxUtil: dropdim1, dropdim2
using Distributions, PDMats
import StatsBase: sample, weights
using StatsFuns: logaddexp
using LinearAlgebra
using Flux, NNlib
using Logging


const DETAILED = LogLevel(-500)


@inline is_tilt(d::GMM, tilt_amt::AbstractFloat) = isapprox(tilt_amt, 1.0) ? d : update(d; sigmas = d.sigmas * tilt_amt)
@inline range_j(j::Signed, chunksize::Signed) = UnitRange((j-1)*chunksize + 1, j*chunksize)


# Changed from Adaptive *Mixture* to Adaptive *Multiple* Importance Sampling (Cornuet et al. '09)
function adamult(dGMM::GMM, log_f::Function; max_iter=10, nsmps=1000,
                    IS_tilt=1.0, mle_max_iter=30, final_smps=1000)

    n_total = max_iter*nsmps
    iter_fin = max_iter       # for logging: assume complete all iters (unless terminate early).
    X, U, L = Matrix{T}(undef, d, n_total), Vector{T}(undef, n_total), Vector{T}(undef, n_total)

    Qs = Vector{GMM}(undef, max_iter)
    cQ = is_tilt(dGMM, IS_tilt)  # exponential tilt if reqd

    for j = 1:max_iter
        Qs[j] = cQ

        cX = rand(cQ, nsmps)
        U[range_j(j, nsmps)] = log_f(cX)

        # Calculate log density of CURRENT SAMPLE under MIXTURE OF ALL IS DENSITIES
        cL = logpdf_(Qs[1], cX)
        for j_prev = 2:j
            cL = logaddexp.(cL, logpdf_(Qs[j_prev], cX))   # log(exp(Lprev) + exp(Lcur))
        end

        # Increment log density of ALL PREV SAMPLES under CURRENT MIXTURE DENSITY
        for j_prev = 1:j-1
            L_jprev = logpdf_(cQ, X[:, range_j(j_prev, nsmps)])
            L[range_j(j_prev, nsmps)] = logaddexp.(L[range_j(j_prev, nsmps)], L_jprev)
        end

        # push samples / log proposal density to "stack"
        L[range_j(j, nsmps)] = cL
        X[:, range_j(j, nsmps)] = cX

        # current weights of ALL MC samples so far...
        W = softmax(U[1:j*nsmps] - L[1:j*nsmps])
        Wcur = softmax(U[range_j(j, nsmps)] - L[range_j(j, nsmps)])
        ess = is_eff_ss(Wcur) # .. but only interested in ESS of new particles
        # bson(format("{:s}_{:d}.bson", save_nm, j), dGMM=cQ, S=X[:, 1:j*nsmps], W=W)

        if ess / nsmps > 0.8
            iter_fin = j
            break
        end

        @logmsg DETAILED format("          | +++ AMIS ({:3d}/{:3d}) NOT TERMINATED, " *
                      "ncls={:4d}, ESS={:.2f}", j, max_iter, ncomponents(cQ), ess)

        gmm_iter = max(mle_max_iter - 2*j,5)
        cQ = gmm_fit(X[:, 1:j*nsmps], W, cQ; max_iter=gmm_iter, tol=1e-3, rm_inactive=true)
        cQ = is_tilt(cQ, IS_tilt)  # exponential tilt if reqd
    end

    S, logW = importance_sample(cQ, final_smps, log_f, shuffle=false)
    W = softmax(logW)
    ess = is_eff_ss(W)
    @logmsg DETAILED format("          | +++ AMIS ({:3d}/{:3d}) TERMINATED, " *
                  "ncls={:4d}, ESS={:.2f}", iter_fin, max_iter, ncomponents(cQ), ess)

    return cQ, S, W, ess, iter
end


function adamult_is(S::Matrix{T}, W::Vector{T}, dGMM::GMM, log_f::Function; max_iter=10,
    nsmps=1000, IS_tilt=1.0, mle_max_iter=30, final_smps=1000) where T <: AbstractFloat

    cQ = gmm_fit(S, W, dGMM; max_iter=mle_max_iter, tol=1e-3, rm_inactive=true)
    adamult_is(cQ, log_f; max_iter=max_iter, nsmps=nsmps,
                        IS_tilt=IS_tilt, mle_max_iter=mle_max_iter, final_smps=final_smps)
end

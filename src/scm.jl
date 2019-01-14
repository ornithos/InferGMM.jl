module scm

using LinearAlgebra
using Parameters, Formatting
using AxUtil
using ..gmm
using AxUtil.Flux: make_lt, make_lt_strict, diag0, unmake_lt_strict
using AxUtil.Math: unmake_lt_strict


export score_match_gmm

function _sum_of_arrays!(x1::Array, x::Vector)
    for j in 2:length(x); x1 .+= x[j]; end
    return x1
end

sum_of_arrays!(x::Vector) = _sum_of_arrays!(x[1], x::Vector)
sum_of_arrays(x::Vector) = _sum_of_arrays!(copy(x[1]), x::Vector)
mean_of_arrays!(x::Vector) = sum_of_arrays!(x)/length(x)
mean_of_arrays(x::Vector) = sum_of_arrays(x)/length(x)

function _log_gauss_llh_Linv(X, mu, L)
    d = size(X,1)
    Z = L*(X .- mu)
    exponent = -0.5*sum(Z.^2, dims=1)[:]
    lognormconst = -d*log(2*pi)/2 -0.5*(-2*sum(log.(diag(L))))
    return exponent .+ lognormconst
end


function gmm_llh(X::AbstractMatrix{T}, mus, Ls, pis) where T <: AbstractFloat
    p, n = size(X)
    k = length(pis)

    P = Array{T, 2}(undef, k, n)
    for j = 1:k
        P[j,:] = _log_gauss_llh_Linv(X, mus[j,:], Ls[j]) .+ log(pis[j])
    end
    return AxUtil.Math.logsumexpcols(P)
end


function responsibilities_Linv(X, mus::AbstractArray{T,2}, Ls::AbstractVector, pis::AbstractArray{T, 1}) where T <: AbstractFloat
    softmax(reduce(vcat, [_log_gauss_llh_Linv(X, mus[j,:], Ls[j]) .+ log(pis[j]) for j in 1:size(mus, 1)]'))
end


@inline build_mat(x_lt, x_diag, n_d::Int) = make_lt_strict(x_lt, n_d) + diag0(exp.(x_diag))
@inline build_Ls(LsLT, LsD, n_d) = [build_mat(L, D, n_d) for (L, D) in zip(LsLT, LsD)]
@inline syminv(x) = (u=inv(x); 0.5(u+u'))

function score_match_objective(X::AbstractMatrix, mus::AbstractMatrix, LsLT::AbstractVector,
                                LsD::AbstractVector, ∇log_p::Function)
    u = ∇log_p(X)
    @assert (size(X) == size(u)) "∇log_p function returns a different size to input"
    score_match_objective(X, mus, LsLT, LsD, u)
end

function score_match_objective(X::AbstractMatrix, mus::AbstractMatrix, LsLT::AbstractVector,
                                LsD::AbstractVector, ∇log_p::AbstractArray)
    _internal_score_match_objective(X, mus, LsLT, LsD, ∇log_p)[1]
end


function _internal_score_match_objective(X::AbstractMatrix, mus::AbstractMatrix, LsLT::AbstractVector,
                                LsD::AbstractVector, ∇log_p::AbstractArray)
    k, n_d = ncomponents(q), size(q)
    @assert (size(X) == size(∇log_p)) "∇log_p function returns a different size to input"

    Ls = build_Ls(LsLT, LsD, n_d)
    Λ = [L'L for L in Ls]

    r = responsibilities_Linv(X, mus, Ls, ones(k)/k)

    score = [Λ[j] * ( r[j,:]' .* (mus[j,:] .- X)) for j in 1:k]
    score = sum_of_arrays(score)  # sums over j (components)
    Δ = score - ∇log_p
    return 0.5*sum(Δ .* Δ), Δ, r
end

function _score_match_objective_and_grad(X::AbstractMatrix, mus::AbstractMatrix, LsLT::AbstractVector,
                                LsD::AbstractVector, ∇log_p::Function)
    u = ∇log_p(X)
    @assert (size(X) == size(u)) "∇log_p function returns a different size to input"
    val, Δ, r = _internal_score_match_objective(X, mus, LsLT, LsD, u)
    ∇mu, ∇LT, ∇D = obj_scm_deriv_all(X, mus, LsLT, LsD, Δ, r)
    return val, ∇mu, ∇LT, ∇D
end


function _score_match_objective_and_grad(X::AbstractMatrix, mus::AbstractMatrix, LsLT::AbstractVector,
                                LsD::AbstractVector, ∇log_p::AbstractArray)
    val, Δ, r = _internal_score_match_objective(X, mus, LsLT, LsD, ∇log_p)
    ∇mu, ∇LT, ∇D = obj_scm_deriv_all(X, mus, LsLT, LsD, Δ, r)
    return val, ∇mu, ∇LT, ∇D
end

@with_kw mutable struct scm_opt
    M::Int64 = 5
    epochs::Int64 = 200
    tol::Float = 1e-3
end


# ────────────────────────────────────────────────────────────────────
#                              Time                   Allocations
#                      ──────────────────────   ───────────────────────
#   Tot / % measured:       12.9s / 28.9%           1.04GiB / 100%

#  Section     ncalls     time   %tot     avg     alloc   %tot      avg
#  ────────────────────────────────────────────────────────────────────
#  main             1    3.74s   100%   3.74s   1.04GiB  100%   1.04GiB
#    scm        5.00k    2.71s  72.4%   541μs    648MiB  61.0%   133KiB
#      gradp    5.00k    2.06s  55.0%   411μs    224MiB  21.1%  45.9KiB
#      gradq    5.00k    486ms  13.0%  97.1μs    281MiB  26.5%  57.6KiB
#      obj      5.00k    150ms  4.01%  30.0μs    142MiB  13.4%  29.0KiB
#    IS         5.00k    590ms  15.8%   118μs    332MiB  31.3%  68.1KiB
#    adam       5.00k    291ms  7.79%  58.2μs   26.2MiB  2.47%  5.36KiB
#    resmp p    5.00k   24.0ms  0.64%  4.81μs   6.94MiB  0.65%  1.42KiB
#  ────────────────────────────────────────────────────────────────────

function score_match_gmm(q::GMM, log_p::Function, ∇log_p::Function;
    M::Int64=5, Mq::Int64=M, Mproposal::Int64=50, epochs::Int64=200, tol::Float64=1e-3)

    k, n_d = ncomponents(q), size(q)

    # Calculate Linv = L^{-1} (Linvs) s.t. LL' = Σ, and ∴ Linv'Linv = Λ
    # We parameterise with the LT cholesky factor and a positive diagonal.
    Linvs = [inv(Matrix(cholesky(q.sigmas[:,:,j]).L)) for j in 1:k]
    # (make Flux versions first so we can use their optim library, and then
    #  make pointers to the memory for use in the main function.)
    pLsLT = [param(unmake_lt_strict(L, n_d)) for L in Linvs]
    pLsD = [param(log.(diag(L))) for L in Linvs]
    pMus = param(q.mus)
    pars = Flux.params(pLsLT..., pLsD..., pMus)
    optim = Flux.ADAM(pars, 1e-2)

    LsLT = [Tracker.data(L) for L in pLsLT]
    LsD = [Tracker.data(D) for D in pLsD]
    mus = Tracker.data(pMus)

    history = zeros(epochs)
    # ===> INNER LOOP <========
    for ee in 1:epochs
        total_obj_val = 0.
        for comp in 1:k

            Linvs = build_Ls(LsLT, LsD, n_d)

            # CHECK THIS DOES THE RIGHT THING
            ϵ = randn(n_d, Mproposal)
            X = mus[comp,:] .+ Linvs[comp] \ ϵ

            # Importance weights and resampling
            logW = log_p(X) - gmm_llh(X, mus, Linvs, ones(k)/k)

            Xq = X[:,1:Mq]
            Xp = X[:,AxUtil.Random.multinomial_indices_linear(M, softmax(logW))]
            X = hcat(Xq, Xp)

            # calc obj and grad
            val, ∇mu, ∇LT, ∇D = _score_match_objective_and_grad(X, mus, LsLT, LsD, ∇log_p)   # pis are currently uniform
            total_obj_val += val

            [(pLsLT[j].grad .= ∇LT[j]) for j in 1:k]
            [(pLsD[j].grad .= ∇D[j]) for j in 1:k]
            pMus.grad .= ∇mu

            # take gradient step
            optim()
        end
        # admin: save obj value and check convergence
        history[ee] = total_obj_val
        if ee > 1 && abs(diff(history[ee-1:ee])[1]/history[ee-1]) < tol
            Linvs = build_Ls(LsLT, LsD, n_d)
            return GMM(mus, cat([syminv(L'L) for L in Linvs]..., dims=3), ones(k)/k), history[1:ee]
        end
    end
    # <=== INNER LOOP <========

    Linvs = build_Ls(LsLT, LsD, n_d)
    return GMM(mus, cat([syminv(L'L) for L in Linvs]..., dims=3), ones(k)/k), history
end


function obj_scm_deriv_all(X, mus, LsLT, LsD, u)
    @assert size(u) == size(X)
    k, n_d = length(LsD), length(LsD[1])

    Ls = build_Ls(LsLT, LsD, n_d)
    Λ = [Ls[j]'Ls[j] for j in 1:k]

    r = responsibilities_Linv(X, mus, Ls, ones(k)/k)

    score = [Λ[j] * ( r[j,:]' .* (mus[j,:] .- X)) for j in 1:k]
    sum_score = sum_of_arrays(score)
    Δ = sum_score - u
    obj_scm_deriv_all(X, mus, LsLT, LsD, Δ, r)
end

function obj_scm_deriv_all(X, mus, LsLT, LsD, Δ, r)

    Ls = build_Ls(LsLT, LsD, n_d)
    Λ = [Ls[j]'Ls[j] for j in 1:k]

    ∇mu = Vector{Any}(undef, k)
    ∇prec = Vector{Any}(undef, k)

    ϵ = [Λ[j] * Δ for j in 1:k]    # error signal transformed by precision_j
    D = [X .- mus[j,:] for j in 1:k]  # raw distance from mean_j
    d = [sum(D[j] .* ϵ[j], dims=1)[:] for j in 1:k]   # distance (raw dist --> error signal) wrt norm prec_j
    ω = [r[j,:] .* d[j] for j in 1:k]  # responsibility weighted distance
    ω_all = sum_of_arrays(ω)
    ξ = [r[j,:] .* ω_all for j in 1:k] # responsibility weighted sum of resp. wgt distances.
    γ = [r[j,:]' .* ϵ[j] for j in 1:k]

    for s in 1:k
        ∇mu[s] = sum(γ[s], dims=2) - Λ[s] * sum(D[s] .*(ω[s] - ξ[s])', dims=2)
        term1 = 0.5*(sum(ξ[s]) - sum(ω[s]))*inv(Λ[s])
        term2 = -0.5*((ξ[s] - ω[s])'.*D[s])*D[s]'
        term3 = -Δ*(r[s,:]' .* D[s])'
        ∇prec[s] = term1 + term2 + term3
    end

    ∇mu = Matrix(reduce(hcat, ∇mu)')
    ∇LsLT = Vector{Any}(undef, k)
    ∇LsD = Vector{Any}(undef, k)

    for j in 1:k
        Lgrad = Ls[j]*(∇prec[j] + ∇prec[j]')
        ∇LsLT[j] = unmake_lt_strict(Lgrad, n_d)
        ∇LsD[j] = diag(Lgrad) .* exp.(LsD[j])
    end
    return ∇mu, ∇LsLT, ∇LsD
end


# NOTE THAT sampling / resampling from p +/or q is analogous
# TO USING MINIBATCHES VS BATCH: GRAD INFORMATION IN A FEW EXAMPLES => IS WASTEFUL
# TO CALC GRADIENT ON ALL.



# TIMEROUTPUTS THE DIFFERENT PARTS OF THE FUNCTION(S).
# CAN
#    * MOVE TO A x'Ae APPROACH IN THE MU DERIVATIVE AS USED IN THE PRECISION
#      WHICH PERFORMS THE MATMULS IN A DIFFERENT ORDER REQING MORE MEMORY
#    * PUSH THEM ALL INTO THE SAME FUNCTION, AVOIDING NEED TO CALC THE INVERSE
#      AND GMM SCORE FUNCTIONS AND RESPONSIBILITIES IN EACH.
#    * ???


# RE-USE PREVIOUSLY CALCULATED VALUES OF THE GRADIENT IF THAT IS THE BOTTLENECK,
# A LITTLE LIKE ADAPTIVE MULTIPLE IS.

# LOOK AT GAUSS NEWTON OPTIMISATION


# WRITE UP MATH.

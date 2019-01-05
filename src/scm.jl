function _sum_of_arrays!(x1::Array, x::Vector)
    for j in 2:length(x); x1 .+= x[j]; end
    return x1
end

sum_of_arrays!(x::Vector) = _sum_of_arrays!(x[1], x::Vector)
sum_of_arrays(x::Vector) = _sum_of_arrays!(copy(x[1]), x::Vector)
mean_of_arrays!(x::Vector) = sum_of_arrays!(x)/length(x)
mean_of_arrays(x::Vector) = sum_of_arrays(x)/length(x)



function score_match_objective(q::GMM, X::AbstractArray, ∇log_p::Function)
    k = ncomponents(q)
    u = log_p(X)
    @assert (size(X) == size(u)) "∇log_p function returns a different size to input"

    r = responsibilities(X, q)
    score = [q.sigmas[:,:,j] \ ( r[j,:]' .* (q.mus[j,:] .- X)) for j in 1:k]
    score = sum_of_arrays(score)  # sums over j (components)
    Δ = score - u
    return 0.5*sum(Δ .* Δ)
end



function _scm_deriv_mu(q::GMM, X::AbstractArray, u::AbstractArray)
    # => EFFICIENCY CAN BE IMPROVED PROBABLY
    k = ncomponents(q)
    Σinvs = [inv(q.sigmas[:,:,j]) for j in 1:k]

    r = responsibilities(X, q)

    score = [Σinvs[j] * ( r[j,:]' .* (q.mus[j,:] .- X)) for j in 1:k]
    unwgt_score = [Σinvs[j] * ((q.mus[j,:] .- X)) for j in 1:k]
    sum_score = sum_of_arrays(score)
    E = sum_score - u

    ∇ = Vector{Any}(undef, k)
    for s in 1:k
        term1 = - score[s] * sum(unwgt_score[s] .* E, dims=1)'
        term2 = Σinvs[s] * sum(r[s,:]' .* E, dims=2)
        cov_correction = term1 + term2
        term3 = score[s] * sum(sum_score .* E, dims=1)'
        ∇[s] = cov_correction + term3
    end

    return Matrix(reduce(hcat, ∇)')
end



function _scm_deriv_prec(q::GMM, X::AbstractArray, u::AbstractArray)
    k = ncomponents(q)
    Σinvs = [inv(q.sigmas[:,:,j]) for j in 1:k]

    r = responsibilities(X, q)

    score = [Σinvs[j] * ( r[j,:]' .* (q.mus[j,:] .- X)) for j in 1:k]
    sum_score = sum_of_arrays(score)
    E = sum_score - u

    ∇ = Vector{Any}(undef, k)

    ϵ = [Σinvs[j] * E for j in 1:k]    # error signal transformed by precision_j
    D = [X .- q.mus[j,:] for j in 1:k]  # raw distance from mean_j
    d = [sum(D[j] .* ϵ[j], dims=1)[:] for j in 1:k]   # distance (raw dist --> error signal) wrt norm prec_j
    ω = [r[j,:] .* d[j] for j in 1:k]  # responsibility weighted distance
    ω_all = sum_of_arrays(ω)
    ξ = [r[j,:] .* ω_all for j in 1:k] # responsibility weighted sum of resp. wgt distances.

    for s in 1:k
        term1 = 0.5*(sum(ξ[s]) - sum(ω[s]))*Σs[s]
        term2 = -0.5*((ξ[s] - ω[s])'.*D[s])*D[s]'
        term3 = -E*(r[s,:]' .* D[s])'
        ∇[s] = term1 + term2 + term3
    end

    return cat(∇... dims=3)
end


# TEST SCRIPT
Random.seed!(42)
dummy_∇logp(x) = sin.((x ./2).^2)
q = GMM...
# ADD UPDATE FOR GMM WITHIN NUM_GRAD
# CHANGE ARGUMENTS TO SCM AND DERIVATIVE
for ii in 1:100
    _x = randn(2, ii)
    Δ = norm(AxUtil.Math.num_grad(x -> score_match_objective(x, _x, _u[:,1:ii]), _mus) - _scm_deriv_mu(_mus, _x, _u[:,1:ii]))
    (Δ > 1e-5) && printfmtln("{:d}: {:.4e}", ii,  Δ)
end



# DO SIMILAR FOR PRECISION MATRIX.



# ADD FINAL STEPS FOR PRECISION MATRIX --> L AND exp(D).




# TEST ON ACTUAL FUNCTION.




# CONSIDER MAKING SCORE MATCHING UNWEIGHTED --> CAN ALWAYS RESAMPLE => NO POINT
# TAKING GRADIENT OF 1000s OF TEST SAMPLES WITH ALMOST NO WEIGHT. WASTEFUL.
# MAY AS WELL RESAMPLE AND WEIGHT UNTIL NEXT ITERATION/EPOCH. THIS IS ANALOGOUS
# TO USING MINIBATCHES VS BATCH: GRAD INFORMATION IN A FEW EXAMPLES => IS WASTEFUL
# TO CALC GRADIENT ON ALL.


# PLACE SCORE MATCHING WITHIN IMPORTANCE SAMPLING SCHEME... SEE HOW IT WORKS.

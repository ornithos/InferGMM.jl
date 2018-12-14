module llh

using LinearAlgebra  #: cholesky, logdet, diag, inv, triu
using AxUtil

export gmm_llh, gmm_llh_invLT

function log_gauss_llh(X, mu, sigma; bypass=false)
    if !bypass
        out = _log_gauss_llh(X, mu, sigma)
    else
        out = - ones(size(X, 2))*Inf
    end
    return out
end


function _log_gauss_llh(X, mu, sigma)
    d = size(X,1)
    invLT = Matrix(inv(cholesky(sigma).L))
    Z = invLT*(X .- mu)
    exponent = -0.5*sum(Z.^2, dims=1)  |> dropdim1
    lognormconst = -d*log(2*pi)/2 -0.5*logdet(sigma)  #.-0.5*(-2*sum(log.(diag(invLT))))
    return exponent .+ lognormconst
end


function _log_gauss_llh_invLT(X, mu, invLT)
    d = size(X,1)
    Z = invLT*(X .- mu)
    exponent = -0.5*sum(Z.^2, dims=1)  |> dropdim1
    lognormconst = -d*log(2*pi)/2 .-0.5*(-2*sum(log.(diag(invLT))))
    return exponent .+ lognormconst
end


function gmm_llh(X, pis, mus, sigmas; thrsh_comp=0.005)
    p, n = size(X)
    k = length(pis)
    inactive_ixs = pis[:] .< thrsh_comp

    P = zeros(k, n)
    for j = 1:k
        P[j,:] = log_gauss_llh(X, mus[j,:], sigmas[:,:,j],
            bypass=inactive_ixs[j]) .+ log(pis[j])
    end
    return AxUtil.Math.logsumexpcols(P)
end

function gmm_llh_invLT(X, pis, mus, invLTs::Array{T,1}; disp=false, thrsh_comp=0.005) where T <: AbstractMatrix
    p, n = size(X)
    k = length(pis)
    inactive_ixs = pis[:] .< thrsh_comp

    P = zeros(k, n)
    for j = 1:k
        if !inactive_ixs[j]
            P[j,:] = _log_gauss_llh_invLT(X, mus[j,:], invLTs[j]) .+ log(pis[j])
        else
            P[j,:] .= -Inf
        end
    end
    return AxUtil.Math.logsumexpcols(P)
end


function responsibilities(X, mus::Matrix{T}, sigmas::Array{T, 3}, pis::Array{T, 1}) where T <: AbstractFloat 
    softmax(reduce(vcat, [log_gauss_llh(X, d.mus[j,:], d.sigmas[:,:,j]) .+ log(d.pis[j]) for j in 1:size(mus, 1)]'))
end

end
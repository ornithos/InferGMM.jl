
# llh_unnorm(Scentered, Linv) = let white = Linv * Scentered; -0.5*sum(white .* white, dims=1); end


# A little like a Gaussian sum approximation for nonlinear dynamical system.
# (oh.. but.. it is an *inner* variational approximation, not an outer one.
# In our case this is not *so* terrible since we do not assume the previous GMM
# is the true posterior as in GS approx., so we do not get catastrophic shrinkage.
# But we should be a little worried.)
@inline function build_mat(x_lt, x_diag, d::Int)
    AxUtil.Flux.make_lt_strict(x_lt, d) + AxUtil.Flux.diag0(exp.(x_diag))
end
@inline _llh_unnorm(Scentered, Linv) = -0.5*sum(x->x*x, Linv * Scentered, dims=1)

# => See AxUtil.Arr
@inline zero_arrays!(x::Array{T, 1}) where T <: AbstractArray = for y in x; zero_arrays!(y); end
@inline zero_arrays!(x::Array{T, 1}) where T <: Real = (x .= 0.);
@inline zero_arrays!(x::Array{T, 2}) where T <: Real = (x .= 0.);

#=======================================================================================
                  Variational Inference (Forward KL) fit of GMM
=======================================================================================#

function optimise_components_bbb(d::GMM, log_f::Function, epochs::Int, batch_size_per_cls::Int;
            converge_thrsh::AbstractFloat=0.999, lr::AbstractFloat=1e-3, auto_lr::Bool=true,
            fix_mean::Bool=false, fix_cov::Bool=false,
            log_f_prev::Union{Function, Nothing}=nothing, anneal_sched::AbstractArray=[1.])
    success = 0
    nanfail = 0   # permit up to 3 in a row failures due to NaN (as this can be from blow-up.)
    haszygote = haskey(Pkg.installed(), "Zygote")
    @assert all( 0. .<= anneal_sched .<= 1.) "annealing schedule array must be ∈ [0,1]^N"
    @assert !xor(log_f_prev==nothing, all(anneal_sched .== 1.)) "both log_f_prev and non-trivial annealing schedule must be specified."

    local history  # local scope of history for persistence outside loop.
    while success < 1
        @debug format("(bbb) LEARNING RATE: {:.3e}", lr)
        # if haszygote
        #     d, history, success = _optimise_components_bbb_zygote(d, log_f, epochs, batch_size_per_cls; converge_thrsh=converge_thrsh, lr=lr, exitifnan=(nanfail<3), auto_lr=auto_lr)
        # else
        d, history, success = _optimise_components_bbb(d, log_f, epochs, batch_size_per_cls;
                                                       converge_thrsh=converge_thrsh,
                                                       lr=lr, exitifnan=(nanfail<3),
                                                       auto_lr=auto_lr,
                                                       fix_mean=fix_mean, fix_cov=fix_cov,
                                                       log_f_prev=log_f_prev, anneal_sched=anneal_sched)
        # end
        lr *= 0.5
        nanfail = (success < 0) * (nanfail - success)   # increment if success = -1, o.w. reset
    end
    return d, history
end

function _failure_dump(ee, dGMM_orig, mupars, invLTpars, invDiagPars)
    display(format("ee = {:d}", ee))
    display("mus originally:")
    display(dGMM_orig.mus)
    display("sigmas originally:")
    display([dGMM_orig.sigmas[:,:,j] for j in 1:size(dGMM_orig.sigmas, 3)])
    display("mu pars:")
    display(mupars.data)
    display("mu grad:")
    display(mupars.grad)
    display("LT pars:")
    display([x.data for x in invLTpars])
    display("LT grad:")
    display([x.grad for x in invLTpars])
    display("Diag pars:")
    display([x.data for x in invDiagPars])
    display("Diag grad:")
    display([x.grad for x in invDiagPars])
end

function _optimise_components_bbb(d::GMM, log_f::Function, epochs::Int, batch_size_per_cls::Int;
        converge_thrsh::AbstractFloat=0.999, lr::AbstractFloat=1e-3, exitifnan::Bool=false,
        auto_lr::Bool=true, fix_mean::Bool=false, fix_cov::Bool=false,
        log_f_prev::Union{Function, Nothing}=nothing, anneal_sched::AbstractArray=[1.])
    # @debug "(bbb) Input GMM: " dGMM=d
    # (ncomponents(d) > 8) && @debug "(bbb) Input GMM: " dGMM=rmcomponents(d, collect(1:8))
    # (ncomponents(d) > 16) && @debug "(bbb) Input GMM: " dGMM=rmcomponents(d, collect(1:16))
    n_d = size(d)
    k = ncomponents(d)
    # if Logging.min_enabled_level(current_logger()) ≤ LogLevel(-2000)
    #     for j = 1:k
    #         @logmsg LogLevel(-500) format("sigma {:d} = ", j) d.sigmas[:,:,j]
    #     end
    # end

    # Set up pars and gradient containers
    invLTpars = [Matrix(inv(cholesky(d.sigmas[:,:,j]).L)) for j in 1:k]  # note that ∵ inverse, Σ^{-1} = L'L
    invDiagPars = [Flux.param(log.(x[diagind(x)])) for x in invLTpars]
    invLTpars = [Flux.param(x[tril!(trues(n_d, n_d), -1)]) for x in invLTpars]

    mupars = Flux.param(d.mus)
    ∇mu = Matrix{partype(d)}(undef, k, n_d)
    ∇Ls = [Matrix{partype(d)}(undef, n_d, n_d) for i in 1:k]
    opt = ADAM(Tracker.Params((mupars, invLTpars..., invDiagPars...)), lr)

    # Admin
    hist_freq = 5   # Sampling freq for history of objective
    s_wdw = 50      # Smoothing window for history for convergence (i.e. s_wdw * hist_freq)
    history = zeros(Int(floor(epochs/hist_freq)))
    s_hist = zeros(Int(floor(epochs/hist_freq)))
    n_anneal = something(findlast(anneal_sched .< 1.), 0)

    rng = MersenneTwister()   # to ensure common random variates for reconstruction and entropy grads.
    @showprogress 1 for ee in 1:epochs
        invLT = [build_mat(x_lt, x_dia, n_d) for (x_lt, x_dia) in zip(invLTpars, invDiagPars)]
        objective = 0.

        η = (ee <= n_anneal) ? anneal_sched[ee] : 1.0  # [1 - annealing amount]

        # Take sample from each component and backprop through recon. term of KL
        # ===> FLUX / AD PART OF PROCEDURE ===================
        for j in 1:k
            ee_seed = rand(rng, 1:2^32 - 1)
            Random.seed!(ee_seed)
            ϵ = randn(n_d, batch_size_per_cls)
            x = mupars[j,:] .+ inv(invLT[j])*ϵ
            objective_j = - η * sum(log_f(x))/batch_size_per_cls  # reconstruction

            Tracker.back!(objective_j)  # accumulate gradients
            objective += objective_j.data  # for objective value
            # ====================================================

            # ---- Annealing (default: none) ----------------
            if η < 1.0
                x = mupars[j,:] .+ inv(invLT[j])*ϵ
                objective_j = - (1-η) * sum(log_f_prev(x))/batch_size_per_cls  # reconstruction

                Tracker.back!(objective_j)  # accumulate gradients
                objective = objective + objective_j.data  # for objective value
            end
            # -----------------------------------------------

            # ===> Calculate entropy of GMM (and gradient thereof)
            Random.seed!(ee_seed)   # common r.v.s (variance reduction)
            _obj = _gmm_entropy_and_grad!(mupars.data, [x.data for x in invLT], ∇mu, ∇Ls; M=batch_size_per_cls, ixs=[j])
            objective += _obj

            mupars.grad .+= ∇mu
            fix_mean && (mupars.grad .= 0.)

            if any(isnan.(mupars.grad))
                display("FAILURE IN MU GRAD")
                _failure_dump(ee, d, mupars, invLTpars, invDiagPars)
                # if exitifnan
                #     return d, -1
                # end
            end

            # convert gradient of ∇L --> gradient of ltri and logdiag components.
            for s = 1:k
                invLTpars[s].grad .+= ∇Ls[s][tril!(trues(n_d, n_d), -1)] # extract lowertri elements
                invDiagPars[s].grad .+= (∇Ls[s][diagind(∇Ls[s])]  .* exp.(invDiagPars[s].data))

                fix_cov && (invLTpars[s].grad .= 0.; invDiagPars[s].grad .= 0.)

                if any(isnan.(invLTpars[s].grad)) || any(isnan.(invDiagPars[s].grad))
                    display(format("FAILURE IN INVLT GRAD {:d}", s))
                    _failure_dump(ee, d, mupars, invLTpars, invDiagPars)
                    # if exitifnan
                    #     return d, -1
                    # end
                end
            end
            # ====================================================


            # ==> [DEBUGGING] check for problems, and dump a bunch of data if so.
            if isnan(objective)
                @warn "objective is NaN"
                _failure_dump(ee, d, mupars, invLTpars, invDiagPars)
                if exitifnan
                    return d, history, -1
                end
            end
            # ====================================================

            opt()  # Perform gradient step / zero grad.
        end
        # Objective and convergence
        if ee % hist_freq == 0
            @debug format("(bbb) ({:3d}/{:3d}), objective: {:.3f}", ee, epochs, objective)
            c_ix = ee÷hist_freq
            history[c_ix] = objective
            # If lr too high, objective explodes: exit
            if auto_lr && c_ix > 20 && ((history[c_ix] - history[c_ix-1]) > 10 * std(history[(c_ix-20):(c_ix-10)]))
                @warn "possible exploding objective: restarting with lower lr."
                @debug format("(bbb) ({:3d}/{:3d}) FAIL. obj_t {:.3f}, obj_t-1 {:.3f}", ee, epochs, history[c_ix], history[c_ix-1])
                return d, history, false
            end
            # Capture long term trend in stochastic objective
            if ee > s_wdw*hist_freq
                # Moving window
                s_hist[(c_ix - s_wdw+1)] = mean(history[(c_ix - s_wdw+1):(c_ix)])
                if ee > 3*s_wdw*hist_freq &&
                    s_hist[(c_ix- s_wdw+1)] > converge_thrsh*s_hist[(c_ix - 3*s_wdw+1)]
                    history = history[1:c_ix]
                    break
                end
            end
        end
    end
    μs = Tracker.data(mupars)
    Σs = zeros(partype(d), n_d, n_d, k)

    for j in 1:k
        Σs[:,:,j] = let xd=Tracker.data(build_mat(invLTpars[j], invDiagPars[j], n_d)); s=inv(xd'xd); (s+s')/2; end
    end

    d_out = GMM(μs, Σs, ones(k)/k)
    return d_out, history, true
end


function _gmm_entropy_and_grad!(mupars::Matrix{T}, invLT::Array, ∇mu::Matrix{T}, ∇L::Array; M::Int=1, ixs=nothing) where T <: AbstractFloat
    k, n_d = size(mupars)
    normcnst = [sum(log.(diag(invLT[i]))) for i in 1:k]

    Ls = invLT
    Linvs = [inv(x) for x in invLT]
    precs = [L'L for L in Ls]

    ∇mu .= 0
    for i in 1:k
        ∇L[i] .= 0
    end
    objective = 0.

    ixs = something(ixs, 1:k)
    for c in ixs
        x = mupars[c,:] .+ Linvs[c]*randn(n_d, M)

        log_q = Array{T, 2}(undef, k, M)
        for j in 1:k
            log_q[j,:] = _llh_unnorm(x .- mupars[j,:], invLT[j]) .+ normcnst[j]
        end
        R, _mllh = AxUtil.Math.softmax_lse(log_q)
        objective += sum(_mllh)/M - log(k)/M
        Mcj = sum(R, dims=2)  # effective number of points from each cluster j for sample from c

        # Calculate ∇μ, ∇L
        for j in 1:k
            if j == c
                ∇L[c] += Linvs[c]' * Mcj[j] / M
                continue
            end
            @views RXmMu = R[j:j, :] .* (x .- mupars[j, :])

            μ_term = mean(precs[j] * RXmMu, dims=2)
            ∇mu[j,:] += μ_term
            ∇mu[c,:] -= μ_term

            # ===== more difficult ∇L terms ============
            # (s == c) update component c for L_c in MC simulation
            @views ∇L[c] += Linvs[c]' * precs[j] * (RXmMu * (x .- mupars[c, : ])')/M # L_c^{-T}*L_j'L_j(x-μ_j)(x-μ_c)^T
            # (s == j) update component j for factors in likelihood
            @views ∇L[j] += (Mcj[j] * Linvs[j]' - Ls[j] * RXmMu * (x .- mupars[j,:])')/M # L^{-T} - L(x-μ)(x-μ)^T
        end
    end
    return objective
end

function _gmm_entropy_and_grad(mupars::Matrix{T}, invLT::Array; M::Int=1, ixs=nothing) where T <: AbstractFloat
    k, n_d = size(mupars)
    ∇mu = Matrix{T}(undef, k, n_d)
    ∇L = [Matrix{T}(undef, n_d, n_d) for i in 1:k]
    objective = _gmm_entropy_and_grad!(mupars, invLT, ∇mu, ∇L; M=M, ixs=ixs)

    return objective, ∇mu, ∇L
end



#=======================================================================================
                  Assumed Density fit (Reverse KL) of GMM
=======================================================================================#


function optimise_components_bbb_revkl(d::GMM, log_f::Function, epochs::Int, batch_size_per_cls::Int;
        converge_thrsh::AbstractFloat=0.999, lr::AbstractFloat=1e-3, exitifnan::Bool=false, auto_lr::Bool=true,
        ixs=nothing, reference_gmm=nothing)
    n_d = size(d)
    k = ncomponents(d)
    ixs = something(ixs, 1:k)

    # Set up parameters for optimisation (NOTE: not using AD, but using implementation of ADAM in Flux)
    invLTval = [Matrix(inv(cholesky(d.sigmas[:,:,j]).L)) for j in 1:k]  # note that ∵ inverse, Σ^{-1} = L'L
    invDiagPars = [Flux.param(log.(x[diagind(x)])) for x in invLTval]
    invLTPars = [Flux.param(x[tril!(trues(n_d, n_d), -1)]) for x in invLTval]
    mupars = Flux.param(copy(d.mus))
    opt = ADAM(Tracker.Params((mupars, invLTPars..., invDiagPars...)), lr)

    # create data views of the Flux params for use in algo
    mus = mupars.data
    invDiagval = [x.data for x in invDiagPars]
    invLTval = [x.data for x in invLTPars]

    # Allocate memory for gradient arrays
    ∇_mu = zeros(k, n_d)
    ∇_invdiag = [zeros(n_d) for _ in 1:k]
    ∇_invlt = [zeros(Int(n_d *(n_d-1)/2)) for _ in 1:k]

    # Save objective value history: note because using local exploration of global
    # integral, the original estimate will be heavily under-estimated, and obj usually increases.
    hist_freq = 5   # Sampling freq for history of objective
    history = zeros(Int(floor(epochs/hist_freq)))

    for ee in 1:epochs
        objective = 0.

        # Take sample from each component and backprop through recon. term of KL
        for c in ixs

            # zero gradient (rather than reallocate): I THINK THIS IS UNNECESSARY.
            # nesting all within one call seems to take a performance hit:
            #  --> perhaps 3 level function recursion cannot be inlined?
            # zero_arrays!(∇_mu); zero_arrays!(∇_invdiag); zero_arrays!(∇_invlt);

            # useful quantities reqd by objective and gradient.
            normcnst = [sum(invDiagval[r]) for r in 1:k] #.-n_d/2*log(2π)
            invLT = [build_mat(x_lt, x_dia, n_d) for (x_lt, x_dia) in zip(invLTval, invDiagval)]
            invinvLT = [inv(x) for x in invLT]

            # Sample from current approximation and calculate importance weights
            # *IMPORTANT*: we don't want to take gradient of parameters in q: this is
            # similar to EM where the expectation is wrt a *previous* version of the same q.
            x = mus[c,:] .+ inv(invLT[c])*randn(n_d, batch_size_per_cls)
            w = log_f(x) - AxUtil.gmm.gmm_llh_invLT(x, ones(k)/k, mus, invLT) # log imp_wgt
            NNlib.softmax!(w)  # exp(w)/sum(exp(w)) => self normalised imp weights

            # Calculate objective (approx. integral of log GMM density)
            log_q_terms = reduce(vcat, [_llh_unnorm(x .- mus[j,:], invLT[j]) for j in 1:k])
            R, _mllh = AxUtil.Math.softmax_lse(log_q_terms .+ normcnst)
            objective += -dot(_mllh, w)

            # Calculate gradients
            MWcj = sum(R .* w', dims=2)  # effective number of points from each cluster j for sample from c
            precs = [L'L for L in invLT]
            for j in 1:k
                @views RXmMu = R[j:j, :] .* (x .- mus[j, :])
                ∇_mu[j,:] = -sum(w' .* (precs[j] * RXmMu), dims=2)

                wRXmMu = (w' .* RXmMu)
                @views ∇L_term = -(MWcj[j] * invinvLT[j]' - invLT[j] * wRXmMu * (x .- mus[j,:])') # L^{-T} - L(x-μ)(x-μ)^T
                ∇_invlt[j] = ∇L_term[tril!(trues(n_d, n_d), -1)] # extract lowertri elements
                ∇_invdiag[j] = diag(∇L_term) .* exp.(invDiagval[j])
            end

            # tfer gradient for optimisation
            mupars.grad .= ∇_mu
            for j in 1:k
                invLTPars[j].grad .= ∇_invlt[j]
                invDiagPars[j].grad .= ∇_invdiag[j]
            end

            opt()   # take optimisation step (ADAM)
        end

        # Objective and convergence
        if ee % hist_freq == 0
            @debug format("(bbb) ({:3d}/{:3d}), objective: {:.3f}", ee, epochs, objective)
            history[ee÷hist_freq] = objective
        end

    end

    Σs = zeros(partype(d), n_d, n_d, k)
    for j in 1:k
        Σs[:,:,j] = let xd=Tracker.data(build_mat(invLTval[j], invDiagval[j], n_d)); s=inv(xd'xd); (s+s')/2; end
    end
    cgmm = GMM(mus, Σs, ones(k)/k)

    return cgmm, history
end

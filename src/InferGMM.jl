module InferGMM
using Flux, Flux.Tracker
using Flux: ADAM
using StatsFuns: logsumexp
using Distributions
using Formatting
using Random # randperm, MersenneTwister
using LinearAlgebra  #: cholesky, logdet, diag, inv, triu
using NNlib: softmax
using AxUtil # Math, array, Flux extensions
using BSON, Logging, ProgressMeter

include("Misc.jl")
using .Misc

include("llh.jl")
include("gmm.jl")


end
module BRR

using IPFitting, LinearAlgebra, Distributions, Statistics
using PyCall

export do_brr, posterior, maxim_hyper

function do_brr(Ψ, Y, α, β, n)
    m, S = posterior(Ψ, Y, α, β)
    d = MvNormal(m, Symmetric(S))
    c_samples = rand(d, n);
    println("$(e) worked")
    return c_samples
end

function posterior(Ψ, Y, α, β; return_inverse=false)
    S_inv = Symmetric(α * Diagonal(ones(length(Ψ[1,:]))) + (β * Symmetric(transpose(Ψ) * Ψ)))
    S = Symmetric(inv(S_inv))
    m = β * (Symmetric(S)*transpose(Ψ)) * Y
    
    if return_inverse
        return m, Symmetric(S), Symmetric(S_inv)
    else
        return m, Symmetric(S)
    end
end

function maxim_hyper(Ψ, Y)
    BRR = pyimport("sklearn.linear_model")["BayesianRidge"]

    clf = BRR()
    clf.fit(Ψ, Y)

    α = clf.alpha_
    β = clf.lambda_
    
    return α, β
end

function log_marginal_likelihood(Ψ, Y, α, β)
    N, M = size(Ψ)
    
    m, _, S_inv = posterior(Ψ, Y, α, β; return_inverse=true)

    E_D = β * sum((Y - Ψ * m).^2)
    E_W = α * sum(m .^ 2.0)

    score = (M * log(α)) + (N * log(β)) - E_D - E_W - logdet(S_inv) - N * log(2*π)
    
    return 0.5 * score
end

end
module BRR

using IPFitting, LinearAlgebra, Distributions, Statistics
using PyCall

export do_brr, posterior, maxim_hyper

function do_brr(Ψ, Y, α, β, n)
    m, S = posterior(Ψ, Y, α, β)
    #d = MvNormal(m, Symmetric(S))
    for e in reverse([10.0^-i for i in 1:50])
        min_eig_val = minimum(eigvals(Symmetric(S)))
        try
            global d = MvNormal(m, Symmetric(S) - (min_eig_val - e)*I)
            break
        catch
        end
     end
    @show α, β
    c_samples = rand(d, n)
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

    clf = BRR(compute_score=true)
    clf.fit(Ψ, Y)

    α = clf.alpha_
    β = clf.lambda_
    c = clf.coef_
    lml_score = clf.scores_
    
    return α, β, c, lml_score[end]
end

# function log_marginal_likelihood(Ψ, Y, α, β)
#     N, M = size(Ψ)
    
#     m, _, S_inv = posterior(Ψ, Y, α, β; return_inverse=true)

#     E_D = β * sum((Y - Ψ * m).^2)
#     E_W = α * sum(m .^ 2.0)

#     score = (M * log(α)) + (N * log(β)) - E_D - E_W - logdet(S_inv) - N * log(2*π)
    
#     return 0.5 * score
# end

function get_coeff(Ψ, Y, ncoms)
    ARDRegression = pyimport("sklearn.linear_model")["ARDRegression"]
    BRR = pyimport("sklearn.linear_model")["BayesianRidge"]

    nbasis = length(Ψ[1,:])

    clf = ARDRegression()
    clf.fit(Ψ, Y)
    norm(clf.coef_)
    inds = findall(clf.coef_ .!= 0)

    @info("Keeping $(length(inds)) basis functions ($(round(length(inds)/nbasis, digits=2)*100)%)")

    clf = BRR()
    clf.fit(Ψ[:,inds], Y)

    S_inv = clf.alpha_ * Diagonal(ones(length(inds))) + clf.lambda_ * Symmetric(transpose(Ψ[:,inds])* Ψ[:,inds])
    S = Symmetric(inv(S_inv))
    m = clf.lambda_ * (Symmetric(S)*transpose(Ψ[:,inds])) * Y

    d = MvNormal(m, Symmetric(S))
    c_samples = rand(d, ncoms);

    c = zeros(nbasis)
    c[inds] = clf.coef_
    
    k = zeros(nbasis, ncoms)
    for i in 1:ncoms
        _k = zeros(nbasis)
        _k[inds] = c_samples[:,i]
        k[:,i] = _k
    end
    return c, k
end

end
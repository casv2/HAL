module BRR

using IPFitting, LinearAlgebra, Distributions, Statistics

export do_brr, posterior

function do_brr(Ψ, Y, α, β, n)
    m, S = posterior(Ψ, Y, α, β)
    d = MvNormal(m, Symmetric(S))
    c_samples = rand(d, n);
    return c_samples
end

function posterior(Ψ, Y, α, β; return_inverse=false)
    S_inv = α * Diagonal(ones(length(Ψ[1,:]))) + (β * (transpose(Ψ) * Ψ))
    S = pinv(S_inv)
    m = β * (S*transpose(Ψ)) * Y
    
    if return_inverse
        return m, S, S_inv
    else
        return m, S
    end
end

end
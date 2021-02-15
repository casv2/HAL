module BRR

using IPFitting, LinearAlgebra, Distributions, Statistics

export do_brr, posterior, maxim_hyper

function do_brr(Ψ, Y, α, β, n)
    m, S = posterior(Ψ, Y, α, β)
    for e in [10.0^(i) for i in -100:-2]
        try
            d = MvNormal(m, Symmetric(S) - (minimum(eigvals(Symmetric(S))) - e)*I)
            c_samples = rand(d, n);
            println("$(e) worked")
            return c_samples
        catch
            println("$(e) didn't work")
        end
    end
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

function maxim_hyper(Ψ, Y, α0=1e-5, β0=1e-5, max_iter=100, ϵ=1e-3)
    N, M = size(Ψ)
    
    eigvals_0 = eigvals(Symmetric(transpose(Ψ) * Ψ))
    
    β = β0
    α = α0
    
    for i in 1:max_iter
        β_prev = β
        α_prev = α
        
        eigvalues = eigvals_0 * β
        
        m, S, S_inv = posterior(Ψ, Y, α, β, return_inverse=true)
        
        γ =  sum(eigvalues ./ (eigvalues .+ α))
        α = γ / sum(m.^2)
        
        β_inv = 1 / (N - γ) * sum((Y - Ψ*m).^2)
        β = 1 / β_inv
        
        if abs(β_prev - β) < ϵ && abs(α_prev - α) < ϵ
            println("Found after $(i) iterations: α=$(α), β=$(β)")
            return α, β
        end
    end

    println("Found after max iterations $(max_iter): α=$(α), β=$(β)")
    return α, β
end

end
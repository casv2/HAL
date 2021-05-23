module COM

using JuLIP, Distributions, LinearAlgebra
using JuLIP.MLIPs: SumIP
using HMD
using Random

export VelocityVerlet_com, get_com_energy_forces

function VelocityVerlet_com(IP, IPs, B, c_samples, at, dt; τ = 1e-10, var=true)
    V = at.P ./ at.M
    
    F = forces(IP, at)  
    E = energy(IP, at)
    varE, varF = get_com_energy_forces(F, IPs, B, c_samples, at, var=var)
      
    F1 = F - τ*varF
    A = F1 ./ at.M

    set_positions!(at, at.X + (V .* dt) + (.5 * A * dt^2))
    
    F = forces(IP, at)  
    E = energy(IP, at)
    varE, varF = get_com_energy_forces(F, IPs, B, c_samples, at, var=var)

    F2 = F - τ*varF
    nA = F2 ./ at.M

    nV = V + (.5 * (A + nA) * dt)
    set_momenta!(at, nV .* at.M)

    #p = maximum((norm.(varF) ./ (norm.(F) .+ minF)))
    p = maximum((norm.(varF)))

    return at, p
end

function VelocityVerlet_com_langevin(IP, IPs, B, c_samples, at, dt, T; γ=0.02, τ = 1e-10, var=true)
    V = at.P ./ at.M
    V = random_v_update(V, at.M, γ, T, dt)

    F = forces(IP, at)  
    E = energy(IP, at)
    varE, varF = get_com_energy_forces(F, IPs, B, c_samples, at, var=var)
      
    F1 = F - τ*varF
    A = F1 ./ at.M

    set_positions!(at, at.X + (V .* dt) + (.5 * A * dt^2))
    
    F = forces(IP, at)  
    E = energy(IP, at)
    varE, varF = get_com_energy_forces(F, IPs, B, c_samples, at, var=var)

    F2 = F - τ*varF
    nA = F2 ./ at.M

    V = V + (.5 * (A + nA) * dt)
    V = random_v_update(V, at.M, γ, T, dt)
    set_momenta!(at, V .* at.M)

    #p = maximum((norm.(varF) ./ (norm.(F) .+ minF)))
    p = maximum((norm.(varF)))

    return at, p
end

function random_v_update(V, M, gamma, T, dt)
    R = rand(Normal(), (length(M)*3)) |> vecs
    c1 = exp(-gamma*dt)
    c2 = sqrt(1-c1^2)*sqrt.(T ./ M)
    return c1*V + c2 .* R
end

function get_com_energy_forces(F, IPs, B, c_samples, at; var=var)
    #E_shift = energy(Vref, at)

    nIPs = length(IPs)

    #E_b = energy(B, at)
    F_b = forces(B, at)
    
    mean_site_Es, Es = HMD.HAL._get_sites(IPs, at)
    Fs = [sum(c_samples[:,i] .* F_b) for i in 1:nIPs];
    
    varE = sum([ (Es[i] .- mean_site_Es).^2 for i in 1:nIPs])/nIPs

    if var
        varF =  sum([ 2*(Es[i,:] .- mean_site_Es) .* (Fs[i] - F) for i in 1:nIPs])/nIPs
    else
        varF =  ( sum([ 2*(Es[i,:] .- mean_site_Es) .* (Fs[i] - F) for i in 1:nIPs])/nIPs ) / varE
    end
    
    return varE, varF
end

end
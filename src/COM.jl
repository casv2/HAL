module COM

using JuLIP, Distributions, LinearAlgebra
using JuLIP.MLIPs: SumIP
using HAL
using Random

#export VelocityVerlet_com, get_com_energy_forces

# function VelocityVerlet_com(Vref, B, c, k, at, dt; τ = 0.0)
#     varE, varF, meanF = get_com_energy_forces(Vref, B, c, k, at)
#     F = meanF - τ * varF
      
#     P = at.P + (0.5 * dt * F) 

#     set_positions!(at, at.X + (dt*(at.P ./ at.M) ))
#     set_momenta!(at, P)

#     varE, varF, meanF = get_com_energy_forces(Vref, B, c, k, at)
#     F = meanF - τ * varF
      
#     P = at.P + (0.5 * dt * F) 
#     set_momenta!(at, P)

#     #p = maximum((norm.(varF) ./ (norm.(F) .+ minF)))
#     p = maximum((norm.(varF)))

#     return at, p
# end


function VelocityVerlet_com_langevin_br(IP, IPs, at, dt, T; γ=0.02, τ = 0.0, Pr0 = 0.0001, μ=μ)
    varE, varF = get_com_energy_forces(IP, IPs, at)
    F = forces(IP, at) - τ * varF
    #F = forces(IP, at) 
    
    P = at.P + (0.5 * dt * F) 
    P = random_p_update(P, at.M, γ, T, dt)

    set_positions!(at, at.X + (dt*(at.P ./ at.M) ))
    set_momenta!(at, P)

    varE, varF = get_com_energy_forces(IP, IPs, at)
    F = forces(IP, at) - τ * varF
    
    P = at.P + (0.5 * dt * F) 
    P = random_p_update(P, at.M, γ, T, dt)
    set_momenta!(at, P)

    at = barostat(IP, at, Pr0; μ=5e-7)

    return at, mean(norm.(varF))
end


function VelocityVerlet_com_langevin(IP, IPs, at, dt, T; γ=0.02, τ = 0.0)
    varE, varF = get_com_energy_forces(IP, IPs, at)
    F = forces(IP, at) - τ * varF
    #F = forces(IP, at) 
    
    P = at.P + (0.5 * dt * F) 
    P = random_p_update(P, at.M, γ, T, dt)

    set_positions!(at, at.X + (dt*(at.P ./ at.M) ))
    set_momenta!(at, P)

    varE, varF = get_com_energy_forces(IP, IPs, at)
    F = forces(IP, at) - τ * varF
    
    P = at.P + (0.5 * dt * F) 
    P = random_p_update(P, at.M, γ, T, dt)
    set_momenta!(at, P)

    return at, mean(norm.(varF))
end

function VelocityVerlet_com(IP, IPs, at, dt; τ = 0.0)
    varE, varF = get_com_energy_forces(IP, IPs, at)
    F = forces(IP, at) - τ * varF
      
    P = at.P + (0.5 * dt * F) 
    set_positions!(at, at.X + (dt*(at.P ./ at.M) ))
    set_momenta!(at, P)

    varE, varF = get_com_energy_forces(IP, IPs, at)
    F = forces(IP, at) - τ * varF
      
    P = at.P + (0.5 * dt * F) 
    set_momenta!(at, P)
    #p = maximum((norm.(varF) ./ (norm.(F) .+ minF)))
    #p = mean((norm.(varF)))
    return at  
end

function VelocityVerlet_com_Zm(IP, IPs, at, dt, A; τ = 0.0)
    varE, varF = get_com_energy_forces(IP, IPs, at)
    F = forces(IP, at) - τ * varF
      
    P = at.P + (0.5 * dt * F) 
    set_positions!(at, at.X + (dt*(at.P ./ at.M) ))

    C = A/norm(P)
    C = 1e-20
    set_momenta!(at, (1+C)*P)

    varE, varF = get_com_energy_forces(IP, IPs, at)
    F = forces(IP, at) - τ * varF
      
    P = at.P + (0.5 * dt * F) 
    set_momenta!(at, P)
    #p = maximum((norm.(varF) ./ (norm.(F) .+ minF)))
    #p = mean((norm.(varF)))
    #p = get_site_uncertainty(IP, IPs, at)

    return at#, p
end

function _get_site(IP, at)
    nats = length(at)
    Es = [sum([site_energy(V, at, i0) for V in IP.components[2:end]]) for i0 in 1:nats]
    return Es
end

softmax(x) = exp.(x) ./ sum(exp.(x))

function get_site_uncertainty(IP, IPs, at; Freg=0.5)
    
    nIPs = length(IPs)
    F = forces(IP, at)
    Fs = Vector(undef, nIPs)

    @Threads.threads for i in 1:nIPs
        Fs[i] = forces(IPs[i], at)
    end

    dFn = norm.(sum([(Fs[m] - F) for m in 1:length(IPs)])/nIPs)
    Fn = norm.(F)

    #p = mean(dFn ./ (Fn .+ Freg))
    p = softmax(dFn ./ (Fn .+ Freg))

    #return p, mean(Fn)
    return maximum(p), mean(Fn)
end

# function get_site_uncertainty(IP, IPs, at)
#     nIPs = length(IPs)
#     Es = zeros(length(at.at), nIPs)

#     for j in 1:nIPs
#         Es[:,j] = _get_site(IPs[j], at.at)
#     end

#     mean_E = _get_site(IP, at.at)

#     E_diff = mean(abs.(Es .- mean_E), dims=2)

#     return maximum(E_diff)
# end

# function VelocityVerlet_com_langevin(IP, IPs, at, dt, T; γ=0.02, τ = 0.0)
#     varE, varF = get_com_energy_forces(IP, IPs, at)
#     F = forces(IP, at) - τ * varF
      
#     P = at.P + (0.5 * dt * F) 
#     P = random_p_update(P, at.M, γ, T, dt)
#     set_positions!(at, at.X + (dt*(at.P ./ at.M) ))
#     set_momenta!(at, P)
#     varE, varF = get_com_energy_forces(IP, IPs, at)
#     F = forces(IP, at) - τ * varF
      
#     P = at.P + (0.5 * dt * F) 
#     P = random_p_update(P, at.M, γ, T, dt)
#     set_momenta!(at, P)

#     #p = maximum((norm.(varF) ./ (norm.(F) .+ minF)))
#     p = mean((norm.(varF)))

#     return at, p
# end

function random_p_update(P, M, γ, T, dt)
    V = P ./ M
    R = rand(Normal(), (length(M)*3)) |> vecs
    c1 = exp(-γ*dt)
    c2 = sqrt(1-c1^2)*sqrt.(T ./ M)
    V_new = c1*V + c2 .* R
    return V_new .* M
end

function barostat(IP, at, Pr0; μ=5e-7)
    Pr = -tr(stress(IP,at)) / 3 * HAL.MD.GPa
    scl_pres = (1.0 - (μ * (Pr0 - Pr)))
    at = set_cell!(at, scl_pres * at.cell)
    at = set_positions!(at, scl_pres * at.X)
    return at
end

function get_com_energy_forces(IP, IPs, at)
    #E_shift = energy(Vref, at)
    nIPs = length(IPs)

    E = energy(IP, at)
    F = forces(IP, at)

    Fs = Vector(undef, nIPs)
    Es = Vector(undef, nIPs)

    @Threads.threads for i in 1:nIPs
        Es[i] = energy(IPs[i], at) 
        Fs[i] = forces(IPs[i], at)
    end

    varF =  sum([ 2*(Es[i] - E)*(Fs[i] - F) for i in 1:nIPs])/nIPs
    
    #meanE = mean(Es)
    varE = sum([ (Es[i] - E)^2 for i in 1:nIPs])/nIPs
    
    #meanF = mean(Fs)
    
    return varE, varF
end

# function get_com_energy_forces(Vref, B, c, k, at)
#     nIPs = length(k[1,:])
#     #E_shift = energy(Vref, at)
#     E = energy(B, at)
#     F = forces(B, at)

#     E_shift = energy(Vref, at)

#     Es = [(E_shift + sum(k[:,i] .* E)) for i in 1:nIPs];
#     Fs = [sum(k[:,i] .* F) for i in 1:nIPs];

#     meanE = E_shift + sum(c .* E)
#     meanF = sum(c .* F)

#     varE = sum([ (Es[i] - meanE)^2 for i in 1:nIPs])/nIPs
#     varF =  sum([ 2*(Es[i] - meanE)*(Fs[i] - meanF) for i in 1:nIPs])/nIPs
#     #varF =  sum([ 2*(Es[i] - meanE)*(Fs[i] - meanF) for i in 1:nIPs])/nIPs
#     # meanE = (E_shift + sum(c .* E))/nats
#     # varE = sum([ (Es[i] - meanE)^2 for i in 1:nIPs])/nIPs

#     #stdF = sqrt(sum(vcat([vcat(Fs[m]...) .- vcat(meanF...) for m in 1:nIPs]...).^2)/length(nIPs))
#     #meanF = mean(Fs)
    
#     return varE, varF, meanF
# end

# function get_com_energy_forces(F, IPs, B, c_samples, at; var=var)
#     #E_shift = energy(Vref, at)

#     nIPs = length(IPs)

#     #E_b = energy(B, at)
#     F_b = forces(B, at)
    
#     mean_site_Es, Es = HAL.HAL._get_sites(IPs, at)
#     Fs = [sum(c_samples[:,i] .* F_b) for i in 1:nIPs];
    
#     varE = sum([ (Es[i] .- mean_site_Es).^2 for i in 1:nIPs])/nIPs

#     if var
#         varF =  sum([ 2*(Es[i,:] .- mean_site_Es) .* (Fs[i] - F) for i in 1:nIPs])/nIPs
#     else
#         varF =  ( sum([ 2*(Es[i,:] .- mean_site_Es) .* (Fs[i] - F) for i in 1:nIPs])/nIPs ) / varE
#     end
    
#     return varE, varF
# end

end
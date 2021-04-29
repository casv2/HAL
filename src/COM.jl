module COM

using JuLIP, Distributions, LinearAlgebra
using JuLIP.MLIPs: SumIP

export VelocityVerlet_com, get_com_energy_forces

function VelocityVerlet_com(IP, Vref, B, c_samples, at, dt; minF=0.5, τ = 1e-10)
    V = at.P ./ at.M
    
    F = forces(IP, at)  
    E = energy(IP, at)
    varE, varF = get_com_energy_forces(E, F, Vref, B, c_samples, at);
      
    F1 = F - τ*varF
    A = F1 ./ at.M

    set_positions!(at, at.X + (V .* dt) + (.5 * A * dt^2))
    
    F = forces(IP, at)  
    E = energy(IP, at)
    varE, varF = get_com_energy_forces(E, F, Vref, B, c_samples, at);

    F2 = F - τ*varF
    nA = F2 ./ at.M

    nV = V + (.5 * (A + nA) * dt)
    set_momenta!(at, nV .* at.M)

    p = maximum((norm.(varF) ./ (norm.(F) .+ minF)))

    return at, p
end

function get_com_energy_forces(E, F, Vref, B, c_samples, at)
    E_shift = energy(Vref, at)

    nIPs = length(c_samples[1,:])

    E_b = energy(B, at)
    F_b = forces(B, at)
    
    Es = [E_shift + sum(c_samples[:,i] .* E_b) for i in 1:nIPs];
    Fs = [sum(c_samples[:,i] .* F_b) for i in 1:nIPs];
    
    varE = sum([ (Es[i] - E)^2 for i in 1:nIPs])/nIPs
    varF =  sum([ 2*(Es[i] - E)*(Fs[i] - F) for i in 1:nIPs])/nIPs
    
    return varE, varF
end

end
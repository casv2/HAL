module COM

using JuLIP, Distributions, LinearAlgebra
using JuLIP.MLIPs: SumIP

export VelocityVerlet_com, get_com_energy_forces

function VelocityVerlet_com(Vref, B, c_samples, at, dt; τ = 1e-10)
    V = at.P ./ at.M
    meanE, varE, meanF, varF = get_com_energy_forces(Vref, B, c_samples, at);    
    F1 = meanF + τ*varF
    
    A = F1 ./ at.M

    set_positions!(at, at.X + (V .* dt) + (.5 * A * dt^2))
    meanE, varE, meanF, varF = get_com_energy_forces(Vref, B, c_samples, at)
    F2 = meanF + τ*varF
    nA = F2 ./ at.M
    nV = V + (.5 * (A + nA) * dt)
    set_momenta!(at, nV .* at.M)

    return at, (norm.(varF) ./ norm.(meanF))
end

function get_com_energy_forces(Vref, B, c_samples, at)
    E_shift = energy(Vref, at)

    nIPs = length(c_samples[1,:])

    E = energy(B, at)
    F = forces(B, at)
    
    Es = [E_shift + sum(c_samples[:,i] .* E) for i in 1:nIPs];
    Fs = [sum(c_samples[:,i] .* F) for i in 1:nIPs];
    
    meanE = mean(Es)
    varE = sum([ (Es[i] - meanE)^2 for i in 1:nIPs])/nIPs
    
    meanF = mean(Fs)
    varF =  sum([ 2*(Es[i] - meanE)*(Fs[i] - meanF) for i in 1:nIPs])/nIPs
    
    return meanE, varE, meanF, varF
end

end
module MD

kB = 8.617330337217213e-05 #units
fs = 0.09822694788464063

using JuLIP, Distributions, LinearAlgebra

export Stationary, VelocityVerlet, MaxwellBoltzmann_scale

function MaxwellBoltzmann(at::Atoms, T::Int)
    d = Normal()
    M = reshape(collect(rand(d, 3*length(at.M))), (length(at.M),3))
    M2 = M .* sqrt.(at.M .* T)
    set_momenta!(at, transpose(M2))

    return at
end

function Stationary(at::Atoms)
    p0 = sum(at.P)
    mtot = sum(at.M)
    v0 = p0 / mtot

    momenta = [at.M[i] * v0 for i in 1:length(at.M)]
    set_momenta!(at, at.P - momenta)

    return at
end

function VelocityVerlet(IP::AbstractCalculator, at::Atoms, dt::Float64)
    V = at.P ./ at.M
    A = forces(IP, at) ./ at.M

    set_positions!(at, at.X + (V .* dt) + (.5 * A * dt^2))

    nA = forces(IP, at) ./ at.M
    nV = V + (.5 * (A + nA) * dt)
    set_momenta!(at, nV .* at.M)

    return at
end

function MaxwellBoltzmann_scale(at::Atoms, T::Int)
    d = Normal()
    M = reshape(collect(rand(d, 3*length(at.M))), (length(at.M),3))

    N = []

    for j in 1:1000
        d = Normal()
        M = reshape(collect(rand(d, 3*length(at.M))), (length(at.M),3))
        n = 0
        for i in 1:1:length(M[:,1])
            n += norm(M[i,:])
        end
        push!(N,n)
    end

    d = Normal()
    M = reshape(collect(rand(d, 3*length(at.M))), (length(at.M),3))
    Mnm = mean(N)

    for i in 1:100000
        n = 0
        for i in 1:length(M[:,1])
            n += norm(M[i,:])
        end
        if n < Mnm
            M = M*1.0001
        else
            M = M*0.9999
        end
    end

    M2 = M .* sqrt.(at.M .* (T * kB))
    set_momenta!(at, transpose(M2))

    return at
end

end

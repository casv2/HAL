using IPFitting
using HMD
using JuLIP
using ACE
using PyCall
using LinearAlgebra
using Random
using Plots
using JuLIP.MLIPs: SumIP
using Statistics
using Distributions

BRR = pyimport("sklearn.linear_model")["BayesianRidge"]

clf = BRR()

al_in = IPFitting.Data.read_xyz("/Users/Cas/Work/ACE/W/tungsten_database.xyz", auto=true)
dia_configs = filter(at -> configtype(at) == "gamma_surface", al_in)
#test_configs = filter(at -> configtype(at) in ["gamma_surface"], al_in)
#test_configs[1000].D["F"]

dia_configs = shuffle!(dia_configs)

R = minimum(IPFitting.Aux.rdf(dia_configs, 4.0))

r0 = rnn(:W)

N=3
deg_site=16
#deg_pair=3

train_ind = convert(Int, length(dia_configs)*0.8)

Bsite = rpi_basis(species = :W,
      N = N,                       # correlation order = body-order - 1
      maxdeg = deg_site,            # polynomial degree
      r0 = r0,                      # estimate for NN distance
      rin = R, rcut = 6.0,   # domain for radial basis (cf documentation)
      pin = 2)                     # require smooth inner cutoff

length(Bsite)

dB = IPFitting.Lsq.LsqDB("", Bsite, dia_configs[1:50])

weights = Dict(
    "default" => Dict("E" => 15.0, "F" => 1.0 , "V" => 1.0 ),
  )

#E0 = -158.54496821
#Vref = OneBody(:Si => E0)
Vref = OneBody(:W => -9.19483512529700)

Ψ, Y = IPFitting.Lsq.get_lsq_system(dB, verbose=true,
                                Vref=Vref, Ibasis = :,Itrain = :,
                                weights=weights, regularisers = [])

clf.fit(Ψ, Y)

S_inv = clf.alpha_ * Diagonal(ones(length(Ψ[1,:]))) + clf.lambda_ * Symmetric(transpose(Ψ)* Ψ)
S = Symmetric(inv(S_inv))
m = clf.lambda_ * (Symmetric(S)*transpose(Ψ)) * Y

d = MvNormal(m, Symmetric(S))
c_samples = rand(d, 100);

IP = SumIP(Vref, JuLIP.MLIPs.combine(dB.basis, c_samples[:,2]))

add_fits_serial!(IP, dia_configs[1:50], fitkey="IP2")
rmse_, rmserel_ = rmse(dia_configs[1:50]; fitkey="IP2");
rmse_table(rmse_, rmserel_)

add_fits_serial!(IP, dia_configs[50:100], fitkey="IP2")
rmse_, rmserel_ = rmse(dia_configs[50:100]; fitkey="IP2");
rmse_table(rmse_, rmserel_)

nIPs = 100

Pl = []
Fl = []

for at in dia_configs[1:1000]
    E_shift = energy(Vref, at.at)

    E = energy(Bsite, at.at)
    F = forces(Bsite, at.at)
        
    Es = [E_shift + sum(c_samples[:,i] .* E) for i in 1:nIPs];
    Fs = [sum(c_samples[:,i] .* F) for i in 1:nIPs];
        
    meanE = mean(Es)
    varE = sum([ (Es[i] - meanE)^2 for i in 1:nIPs])/nIPs
        
    meanF = mean(Fs)
    varF =  sum([ 2*(Es[i] - meanE)*(Fs[i] - meanF) for i in 1:nIPs])/nIPs

    p = maximum(norm.(varF) ./ norm.(meanF))
    push!(Pl,p)

    IP = SumIP(Vref, JuLIP.MLIPs.combine(dB.basis, c_samples[:,1]))
    f = sqrt(mean((vcat(forces(IP, at.at)...) - at.D["F"]).^2))
    push!(Fl, f)
end

Fl

maximum(Fl)
maximum(Pl)

using LaTeXStrings

scatter(Pl[50:end], Fl[50:end], yscale=:log, xscale=:log, label="test")
scatter!(Pl[1:50], Fl[1:50], yscale=:log, xscale=:log, legend=:bottomright, label="training")
xlabel!(L"\max (\| F_{\sigma} \| / \|  F \|)")
ylabel!("RMSE Force Error [eV/A]")
savefig("W_gamma_surface_forces.pdf")
norm.(varF)

norm.(meanF)

(std(Fs) ./ mean(Fs)) * 100

















non_zero_ind = findall(x -> x != 0.0, c)
zero_ind = findall(x -> x == 0.0, c)

# Ψred = Ψ[:, setdiff(1:end, zero_ind)]

# α, β = HMD.BRR.maxim_hyper(Ψred, Y)

# m, S = HMD.BRR.posterior(Ψred, Y, α, β)

zero_ind

Us = []
Es = []
Us1 = []
Es1 = []

IP = SumIP(Vref, JuLIP.MLIPs.combine(dB.basis, c))

for at in al_in#md_configs[train_ind:end]
    u = dot(energy(Bsite, at.at)[setdiff(1:end, zero_ind)], S * energy(Bsite, at.at)[setdiff(1:end, zero_ind)])
    e = (energy(IP, at.at) - at.D["E"][1])/length(at.at)
    push!(Us, u)
    push!(Es, e)
end

scatter(Us .+ 1E-6, abs.(Es) .+ 1E-6, xscale=:log, yscale=:log, legend=:bottomright,label="test")
#scatter!(Us1, abs.(Es1), xscale=:log, yscale=:log, label="training")
xlabel!("Uncertainty Estimate")
ylabel!("Energy Error [eV/atom]")
savefig("HAL_error.pdf")


Us = []
Fs = []
Us1 = []
Fs1 = []

for at in test_configs
    u = dot(energy(Bsite, at.at)[setdiff(1:end, zero_ind)], S * energy(Bsite, at.at)[setdiff(1:end, zero_ind)])
    f = sqrt(mean((vcat(forces(IP, at.at)...) - at.D["F"]).^2))
    push!(Us, u)
    push!(Fs, f)
end

scatter(vcat(Us...) .+ 1E-6, abs.(vcat(Fs...)) .+ 1E-6, yscale=:log, xscale=:log)
ylims!(0.2E-2, 10E2)
xlims!(0.05E-5, 10E3)

#vcat(Fs...)

# for at in al
#     u = dot(energy(Bsite, at.at)[setdiff(1:end, zero_ind)], S * energy(Bsite, at.at)[setdiff(1:end, zero_ind)])
#     e = (energy(IP, at.at) - at.D["E"][1])/length(at.at)
#     push!(Us1, u)
#     push!(Es1, e)
# end

c = [1,0,2]
filter!(x -> x != 0, c)
c
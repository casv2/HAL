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


al_in = IPFitting.Data.read_xyz("/Users/Cas/Work/ACE/Si/Si.xyz", energy_key="dft_energy", force_key="dft_force")
dia_configs = filter(at -> configtype(at) == "dia", al_in)
#test_configs = filter(at -> configtype(at) in ["gamma_surface"], al_in)
#test_configs[1000].D["F"]

dia_configs = shuffle!(dia_configs)

R = minimum(IPFitting.Aux.rdf(dia_configs, 4.0))

r0 = rnn(:W)

N=3
deg_site=16
#deg_pair=3

#train_ind = convert(Int, length(dia_configs)*0.8)

Bsite = rpi_basis(species = :Si,
      N = N,                       # correlation order = body-order - 1
      maxdeg = deg_site,            # polynomial degree
      r0 = r0,                      # estimate for NN distance
      rin = R, rcut = 6.0,   # domain for radial basis (cf documentation)
      pin = 2)                     # require smooth inner cutoff

length(Bsite)

dB = IPFitting.Lsq.LsqDB("", Bsite, dia_configs[1:20])

weights = Dict(
    "default" => Dict("E" => 15.0, "F" => 1.0 , "V" => 1.0 ),
  )

E0 = -158.54496821
Vref = OneBody(:Si => E0)
#Vref = OneBody(:W => -9.19483512529700)

Ψ, Y = IPFitting.Lsq.get_lsq_system(dB, verbose=true,
                                Vref=Vref, Ibasis = :,Itrain = :,
                                weights=weights, regularisers = [])

Ψ
BRR = pyimport("sklearn.linear_model")["ARDRegression"]

clf = BRR(threshold_lambda=200000)
clf.fit(Ψ, Y)
norm(clf.coef_)
inds = findall(clf.coef_ .!= 0)

BRR = pyimport("sklearn.linear_model")["BayesianRidge"]

clf = BRR()
clf.fit(Ψ[:,inds], Y)

S_inv = clf.alpha_ * Diagonal(ones(length(inds))) + clf.lambda_ * Symmetric(transpose(Ψ[:,inds])* Ψ[:,inds])
S = Symmetric(inv(S_inv))
m = clf.lambda_ * (Symmetric(S)*transpose(Ψ[:,inds])) * Y

d = MvNormal(m, Symmetric(S))
c_samples = rand(d, 100);

k = zeros(length(Bsite))
k[inds] = clf.coef_

IP = SumIP(Vref, JuLIP.MLIPs.combine(dB.basis, k))
# nIPs = 100

# at = dia_configs[100].at


# F = forces(Bsite, at)

# Fs = [sum(c_samples[:,i] .* F) for i in 1:nIPs];

# mean(Fs) - forces(IP, at)
# ###
# varF = sqrt.(sum([(Matrix(hcat(Fs[i]...)) .- Matrix(hcat(meanF...))).^2 for i in 1:nIPs])./nIPs)
# norm.(eachcol(varF)) ./ norm.(meanF)
# ####

add_fits_serial!(IP, dia_configs[20:100], fitkey="IP2")
rmse_, rmserel_ = rmse(dia_configs[20:100]; fitkey="IP2");
rmse_table(rmse_, rmserel_)

add_fits_serial!(IP, dia_configs[1:20], fitkey="IP2")
rmse_, rmserel_ = rmse(dia_configs[1:20]; fitkey="IP2");
rmse_table(rmse_, rmserel_)

function expand_forces(basis, train; wE = 1.0, wF = 1.0)
    nobs = sum( 3*length(t.at) for t in train )
    Φ = zeros(nobs, length(basis))
    irow = 0
    for at in train
      #irow += 1
      #Φ[irow, :] = wE * energy(basis, at.at) / length(at)

      nf = 3*length(at)
      Fb = forces(basis, at.at)
      for ib = 1:length(basis)
         Φ[(irow+1):(irow+nf), ib] = wF * mat(Fb[ib])[:]
      end
      irow += nf
    end
    return Φ
end

function expand_energy(basis, train; wE = 1.0, wF = 1.0)
    nobs = sum( 1 for t in train )
    Φ = zeros(nobs, length(basis))
    irow = 0
    for at in train
      irow += 1
      Φ[irow, :] = wE * energy(basis, at.at) / length(at)
    end
    return Φ
end

Φf = expand_forces(Bsite, [al_in[2000]])[:, inds]
Φe = expand_energy(Bsite, [al_in[2000]])[:, inds]

Pl = []
Fl = []

S

for (i,at) in enumerate(vcat(dia_configs[1:20], al_in[1:5:end]))
    @show i
    #IP = SumIP(Vref, JuLIP.MLIPs.combine(dB.basis, clf.coef_))

    #F = forces(Bsite, at.at)
    #varF = (1/clf.lambda_) .* sum(((clf.coef_ .* F) * S) .* (clf.coef_ .* F), dims=2)
    Φf = expand_forces(Bsite, [at])[:, inds]
    Φe = expand_energy(Bsite, [at])[:, inds]
    #p = maximum(sum(((Φ*S).*Φ), dims=2))
    p = maximum(sum(((Φe*S).*Φf), dims=2))

    #p = (norm.(varF) ./ norm.(F))
    push!(Pl,maximum(p))

    f = maximum(vcat(forces(IP,at.at)...) .- at.D["F"])
    push!(Fl, f)
end

maximum(Fl)
maximum(Pl)

Pl[1:20]
Fl[1:20]
Fl[20:end]
using LaTeXStrings

scatter(Pl[20:end] .+ 1E-5, Fl[20:end] .+ 1E-5, yscale=:log, xscale=:log, label="test")
scatter!(Pl[1:20] .+ 1E-5, Fl[1:20] .+ 1E-5, yscale=:log, xscale=:log, legend=:bottomright, label="training")
#vline!([5])
ylims!(0.005, 100)
xlabel!(L"\max (\| F_{\sigma} \| / \|  F \|)")
ylabel!(L"\max (\| \Delta F \|) \quad [eV/A]")
savefig("Si_uncertainty_force_error_cov_sp.pdf")
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
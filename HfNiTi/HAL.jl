using IPFitting
using HMD
using JuLIP
using ACE
using PyCall
using LinearAlgebra
using Plots
using JuLIP.MLIPs: SumIP

ARDRegression = pyimport("sklearn.linear_model")["BayesianRidge"]

clf = ARDRegression(compute_score=true)

al = IPFitting.Data.read_xyz("./HfNiTi/HfNiTiXYZs/iteration1/initial1to4.xyz", energy_key="energy", force_key="forces", virial_key="virial")

R = minimum(IPFitting.Aux.rdf(al, 4.0))

r0 = rnn(:Ti)

N=3
deg_site=10
#deg_pair=3

Bsite = rpi_basis(species = [:Hf, :Ti, :Ni],
      N = N,                       # correlation order = body-order - 1
      maxdeg = deg_site,            # polynomial degree
      r0 = r0,                      # estimate for NN distance
      rin = R, rcut = 6.0,   # domain for radial basis (cf documentation)
      pin = 2)                     # require smooth inner cutoff

dB = LsqDB("", Bsite, al)

weights = Dict(
    "default" => Dict("E" => 30.0, "F" => 1.0 , "V" => 0.1 ),
  )

Vref = OneBody(Dict("Hf" => -5.58093, "Ni" => -0.09784, "Ti" => -1.28072))

Ψ, Y = IPFitting.Lsq.get_lsq_system(dB, verbose=true,
                                Vref=Vref, Ibasis = :,Itrain = :,
                                weights=weights, regularisers = [])

clf.fit(Ψ, Y)

c = clf.coef_

S = clf.sigma_

non_zero_ind = findall(x -> x != 0.0, c)
zero_ind = findall(x -> x == 0.0, c)

# Ψred = Ψ[:, setdiff(1:end, zero_ind)]

# α, β = HMD.BRR.maxim_hyper(Ψred, Y)

# m, S = HMD.BRR.posterior(Ψred, Y, α, β)

al2 = IPFitting.Data.read_xyz("./HfNiTi/HfNiTi_DB.xyz", energy_key="energy", force_key="forces", virial_key="virial")

zero_ind

Us = []
Es = []
Us1 = []
Es1 = []

IP = SumIP(Vref, JuLIP.MLIPs.combine(dB.basis, c))

for at in al2
    u = dot(energy(Bsite, at.at)[setdiff(1:end, zero_ind)], S * energy(Bsite, at.at)[setdiff(1:end, zero_ind)])
    e = (energy(IP, at.at) - at.D["E"][1])/length(at.at)
    push!(Us, u)
    push!(Es, e)
end

for at in al
    u = dot(energy(Bsite, at.at)[setdiff(1:end, zero_ind)], S * energy(Bsite, at.at)[setdiff(1:end, zero_ind)])
    e = (energy(IP, at.at) - at.D["E"][1])/length(at.at)
    push!(Us1, u)
    push!(Es1, e)
end


#plot(Us)

plot()
scatter(abs.(Us), abs.(Es), xscale=:log, yscale=:log, legend=:bottomright, label="test")
scatter!(abs.(Us1), abs.(Es1), xscale=:log, yscale=:log, label="training")
xlabel!("Uncertainty Estimate")
ylabel!("Energy Error [eV/atom]")
savefig("HAL_error.pdf")


Us = []
Fs = []
Us1 = []
Fs1 = []


for at in al2
    u = dot(energy(Bsite, at.at)[setdiff(1:end, zero_ind)], S * energy(Bsite, at.at)[setdiff(1:end, zero_ind)])
    f = sqrt(mean((vcat(forces(IP, at.at)...) - at.D["F"]).^2))
    push!(Us, u)
    push!(Fs, f)
end


scatter(abs.(vcat(Us...)), abs.(vcat(Fs...)))

#vcat(Fs...)

# for at in al
#     u = dot(energy(Bsite, at.at)[setdiff(1:end, zero_ind)], S * energy(Bsite, at.at)[setdiff(1:end, zero_ind)])
#     e = (energy(IP, at.at) - at.D["E"][1])/length(at.at)
#     push!(Us1, u)
#     push!(Es1, e)
# end
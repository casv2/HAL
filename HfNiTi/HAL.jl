using IPFitting
using HMD
using JuLIP
using ACE
using PyCall
using LinearAlgebra
using Plots
using JuLIP.MLIPs: SumIP
using Random
using Statistics
using Distributions
using LaTeXStrings

ARDRegression = pyimport("sklearn.linear_model")["ARDRegression"]

R = minimum(IPFitting.Aux.rdf(al, 4.0))

r0 = rnn(:Ti)

N=3
deg_site=8
deg_pair=3

Bpair = pair_basis(species = [:Hf, :Ti, :Ni],
                r0 = r0,
                maxdeg = deg_pair,
                rcut = 7.0,
                pcut = 1,
                pin = 0) 

Bsite = rpi_basis(species = [:Hf, :Ti, :Ni],
      N = N,                       # correlation order = body-order - 1
      maxdeg = deg_site,            # polynomial degree
      r0 = r0,                      # estimate for NN distance
      rin = R, rcut = 6.0,   # domain for radial basis (cf documentation)
      pin = 2)                     # require smooth inner cutoff

B = JuLIP.MLIPs.IPSuperBasis([Bpair, Bsite]);

weights = Dict(
    "default" => Dict("E" => 30.0, "F" => 1.0 , "V" => 0.1 ),
  )

Vref = OneBody(Dict("Hf" => -5.58093, "Ni" => -0.09784, "Ti" => -1.28072))

function get_coeff(al, B, n, weights, Vref)
    ARDRegression = pyimport("sklearn.linear_model")["ARDRegression"]
    BRR = pyimport("sklearn.linear_model")["BayesianRidge"]

    clf = ARDRegression(compute_score=true)
    dB = LsqDB("", B, al)
    Ψ, Y = IPFitting.Lsq.get_lsq_system(dB, verbose=true,
                                Vref=Vref, Ibasis = :,Itrain = :,
                                weights=weights, regularisers = [])

    clf.fit(Ψ, Y)
    norm(clf.coef_)
    inds = findall(clf.coef_ .!= 0)

    clf = BRR()
    clf.fit(Ψ[:,inds], Y)

    S_inv = clf.alpha_ * Diagonal(ones(length(inds))) + clf.lambda_ * Symmetric(transpose(Ψ[:,inds])* Ψ[:,inds])
    S = Symmetric(inv(S_inv))
    m = clf.lambda_ * (Symmetric(S)*transpose(Ψ[:,inds])) * Y

    d = MvNormal(m, Symmetric(S))
    c_samples = rand(d, n);

    c = zeros(length(B))
    c[inds] = clf.coef_
    
    k = zeros(length(B), n)
    for i in 1:n
        _k = zeros(length(B))
        _k[inds] = c_samples[:,i]
        k[:,i] = _k
    end
    return c, k
end

function get_uncertainties_errors(al_test, B, Vref, c, k)
    IP = SumIP(Vref, JuLIP.MLIPs.combine(B, c))

    n = length(k[1,:])
    Pl = []
    El = []
    for (i,at) in enumerate(al_test)
        E = energy(B, at.at)
        E_shift = energy(Vref, at.at)

        Es = [E_shift + sum(k[:,i] .* E) for i in 1:n];

        meanE = mean(Es)
        varE = sum([ (Es[i] - meanE)^2 for i in 1:n])/n
        push!(Pl, varE)

        e = abs.((energy(IP, at.at) .- at.D["E"][1])/length(at.at))
        push!(El, e)
    end
    return El, Pl
end


al = IPFitting.Data.read_xyz("./HfNiTi/HfNiTiXYZs/iteration1/initial1to4.xyz", energy_key="energy", force_key="forces", virial_key="virial")[1:10:end]
al2 = IPFitting.Data.read_xyz("./HfNiTi/HfNiTi_TEST_DB.xyz", energy_key="energy", force_key="forces", virial_key="virial")

#c, k = get_coeff(al, B, 1000, weights, Vref)

for i in 1:5
    c, k = get_coeff(al, B, 1000, weights, Vref)
    El_train, Pl_train = get_uncertainties_errors(al, B, Vref, c, k)
    El_test, Pl_test = get_uncertainties_errors(al_test, B, Vref, c, k)
    scatter(Pl_test, El_test, yscale=:log, xscale=:log, legend=:bottomright, label="test")
    scatter!(Pl_train, El_train, yscale=:log, xscale=:log,label="train")
    xlabel!(L" \sigma^2(x)")
    ylabel!(L" \Delta E  \quad [eV/atom]")
    savefig("HAL_$(i).pdf")

    maxvals = sort(Pl_test)[end-10:end]
    inds = [findall(Pl_test .== maxvals[end-i])[1] for i in 1:10]
    al = vcat(al, al2[inds])
end

length(al)
IP = SumIP(Vref, JuLIP.MLIPs.combine(B, c))
save_dict("./HfNiTi/fit.json", Dict("IP" => write_dict(IP)))


rand(1:48, 20)

p = plot()

maxPs = [] 
Is = []

for i in 1:100
    try
        tr1 = IPFitting.Data.read_xyz("./relax_$(i).xyz", energy_key="energy", force_key="forces", virial_key="virial")
        #El_test1, Pl_test1 = get_uncertainties_errors(tr1, B, Vref, c, k)
        #push!(maxPs, maximum(Pl_test1))
        push!(Is,i)
        #plot!(p, Pl_test1, xscale=:log, yscale=:log, label="")
    catch
        continue
    end
end

xlabel!("iteration index along relaxation path")
ylabel!(L"\sigma^2(x) ")
savefig("HfNiTi_uncertainty_along_relaxation_path.pdf")
display(p)

maxvals = sort(maxPs)[end-3:end]
inds = [findall(maxPs .== maxvals[end-i])[1] for i in 1:3]
Is[inds]

tr1 = IPFitting.Data.read_xyz("./relax_43.xyz", energy_key="energy", force_key="forces", virial_key="virial")
El_test1, Pl_test1 = get_uncertainties_errors(tr1, Bsite, Vref, c, k)

El_test1

xlabel!("iteration index along relaxation path")
ylabel!(L"\sigma^2(x) ")
savefig("HfNiTi_uncertainty_along_relaxation_path.pdf")
display(p)


savefig("iterative_uncertainty_grades.pdf")
al2[inds]

length(al)

Pl = []
El = []
Fl = []

c, k = get_coeff(al, Bsite, 1000, weights, Vref)

n = 1000

for (i,at) in enumerate(vcat(al, al2))
    E = energy(Bsite, at.at)
    E_shift = energy(Vref, at.at)

    Es = [E_shift + sum(k[:,i] .* E) for i in 1:n];

    meanE = mean(Es)
    varE = sum([ (Es[i] - meanE)^2 for i in 1:n])/n
    push!(Pl, varE)
    #Es = [E_shift + sum(c_samples[:,i] .* E) for i in 1:nIPs];
    
    meanE = mean(Es)
    varE = sum([ (Es[i] - meanE)^2 for i in 1:nIPs])/nIPs
    push!(Pl, varE)

    e = abs.((energy(IP, at.at) .- at.D["E"][1])/length(at.at))
    push!(El, e)

    #F = forces(Bsite, at.at)
    #Fs = [sum(c_samples[:,i] .* F) for i in 1:nIPs];

    #meanF = mean(Fs)
    #varF =  sum([ 2*(Es[i] - meanE)*(Fs[i] - meanF) for i in 1:nIPs])/nIPs
    
    #F = forces(IP, at.at)
    #p = (norm.(varF) ./ norm.(F))
    #push!(Pl,maximum(p))

    #f = maximum(vcat(forces(IP,at.at)...) .- at.D["F"])
    #push!(Fl, f)
end



# maxvals = sort(Pl)[end-5:end]
# inds = [findall(Pl .== maxvals[end-i])[1] for i in 1:5]

# vcat(al, al2)[inds]

# Pl
# Fl

scatter(Pl[60:end] .+ 1E-8, El[60:end] .+ 1E-8, yscale=:log, xscale=:log,label="test")
scatter!(Pl[1:60] .+ 1E-8, El[1:60] .+ 1E-8, yscale=:log, xscale=:log, legend=:bottomright, label="training")
#vline!([5])
#ylims!(0.01, 100)
xlabel!(L" \sigma^2(x)")
ylabel!(L" \Delta E  \quad [eV]")
savefig("HAL_error_iterative.pdf")

IP = SumIP(Vref, JuLIP.MLIPs.combine(dB.basis, c))

add_fits_serial!(IP, al, fitkey="IP2")
rmse_, rmserel_ = rmse(al; fitkey="IP2");
rmse_table(rmse_, rmserel_)

add_fits_serial!(IP, al2, fitkey="IP2")
rmse_, rmserel_ = rmse(al2; fitkey="IP2");
rmse_table(rmse_, rmserel_)
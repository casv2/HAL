module HAL

using JuLIP
using ACE
using IPFitting
using PyCall
using LinearAlgebra
using Plots
using JuLIP.MLIPs: SumIP
using Random
using Statistics
using Distributions
using LaTeXStrings
using ASE

function get_coeff(al, B, ncomms, weights, Vref)
    ARDRegression = pyimport("sklearn.linear_model")["ARDRegression"]
    BRR = pyimport("sklearn.linear_model")["BayesianRidge"]

    clf = ARDRegression()#compute_score=true)
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
    c_samples = rand(d, ncomms);

    c = zeros(length(B))
    c[inds] = clf.coef_
    
    k = zeros(length(B), ncomms)
    for i in 1:ncomms
        _k = zeros(length(B))
        _k[inds] = c_samples[:,i]
        k[:,i] = _k
    end
    return c, k
end

function get_E_uncertainties(al_test, B, Vref, c, k)
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

function get_F_uncertainties(al_test, B, Vref, c, k)
    IP = SumIP(Vref, JuLIP.MLIPs.combine(B, c))

    n = length(k[1,:])
    Pl = []
    Fl = []
    for (i,at) in enumerate(al_test)
        E = energy(B, at.at)
        F = forces(B, at.at)

        E_shift = energy(Vref, at.at)

        Es = [E_shift + sum(k[:,i] .* E) for i in 1:n];
        Fs = [sum(k[:,i] .* F) for i in 1:n];

        meanE = mean(Es)
        varE = sum([ (Es[i] - meanE)^2 for i in 1:n])/n

        meanF = mean(Fs)
        varF =  sum([ 2*(Es[i] - meanE)*(Fs[i] - meanF) for i in 1:n])/n

        F = forces(IP, at.at)
        p = (norm.(varF) ./ norm.(F))
        push!(Pl,maximum(p))

        f = maximum(vcat(F...) .- at.D["F"])
        push!(Fl, f)
    end
    return Fl, Pl
end

function HAL_E(al, al_test, B, ncomms, iters, nadd, weights, Vref)
    for i in 1:iters
        @show("ITERATION $(i)")
        c, k = get_coeff(al, B, ncomms, weights, Vref)

        IP = SumIP(Vref, JuLIP.MLIPs.combine(B, c))

        @show("HAL ERRORS OF ITERATION $(i)")
        add_fits_serial!(IP, al, fitkey="IP2")
        rmse_, rmserel_ = rmse(al; fitkey="IP2");
        rmse_table(rmse_, rmserel_)

        El_train, Pl_train = get_E_uncertainties(al, B, Vref, c, k)
        El_test, Pl_test = get_E_uncertainties(al_test, B, Vref, c, k)
        scatter(Pl_test, El_test, yscale=:log10, xscale=:log10, legend=:bottomright, label="test")
        scatter!(Pl_train, El_train, yscale=:log10, xscale=:log10,label="train")
        xlabel!(L" \sigma^2(x)")
        ylabel!(L" \Delta E  \quad [eV/atom]")
        savefig("HAL_$(i).png")

        Pl_test_fl = filter(!isnan, Pl_test)
        maxvals = sort(Pl_test_fl)[end-nadd:end]

        inds = [findall(Pl_test .== maxvals[end-i])[1] for i in 1:nadd]

        al = vcat(al, al_test[inds])

        save_configs(al_test[inds], i)
        #save_configs(al, i)
    end
end

function HAL_F(al, al_test, B, ncomms, iters, nadd, weights, Vref)
    for i in 1:iters
        c, k = get_coeff(al, B, ncomms, weights, Vref)

        IP = SumIP(Vref, JuLIP.MLIPs.combine(B, c))

        add_fits_serial!(IP, al, fitkey="IP2")
        rmse_, rmserel_ = rmse(al; fitkey="IP2");
        rmse_table(rmse_, rmserel_)

        Fl_train, Pl_train = get_F_uncertainties(al, B, Vref, c, k)
        Fl_test, Pl_test = get_F_uncertainties(al_test, B, Vref, c, k)
        scatter(Pl_test .+ 1E-6, Fl_test .+ 1E-6, yscale=:log10, xscale=:log10, legend=:bottomright, label="test")
        scatter!(Pl_train .+ 1E-6, Fl_train .+ 1E-6, yscale=:log10, xscale=:log10,label="train")
        xlabel!(L"\max (\| F_{\sigma} \| / \|  F \|)")
        ylabel!(L"\max (\| \Delta F \|) \quad [eV/A]")
        savefig("HAL_$(i).png")

        Pl_test_fl = filter(!isnan, Pl_test)
        maxvals = sort(Pl_test_fl)[end-nadd:end]

        inds = [findall(Pl_test .== maxvals[end-i])[1] for i in 1:nadd]

        al = vcat(al, al_test[inds])

        save_configs(al, i)
    end
end

function save_configs(al, i)
    py_write = pyimport("ase.io")["write"]
    al_save = []
    for at in al
        py_at = ASEAtoms(at.at)

        D_info = PyDict(py_at.po[:info])
        D_arrays = PyDict(py_at.po[:arrays])

        D_info["config_type"] = "HAL_$(i)_" * configtype(at)
        try D_info["energy"] = at.D["E"] catch end
        try D_info["virial"] = [at.D["V"][1], at.D["V"][6], at.D["V"][5], at.D["V"][6], at.D["V"][2], at.D["V"][4], at.D["V"][5], at.D["V"][4], at.D["V"][3]] catch end
        try D_arrays["forces"] = reshape(at.D["F"], length(at.at), 3) catch end

        py_at.po[:info] = D_info
        py_at.po[:arrays] = D_arrays

        push!(al_save, py_at.po)
    end
    py_write("HAL_$(i).xyz", PyVector(al_save))
end

end
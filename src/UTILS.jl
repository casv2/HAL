module UTILS

using JuLIP
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
using HAL

function dimer_energy(IP, r::Float64, spec1, spec2)
    X = [[0.0,0.0,0.0], [0.0, 0.0, r]]
    C = [[100.0,0.0,0.0], [0.0, 100.0, 0.0],[0.0, 0.0, 100.0] ]
    at = Atoms(X, [[0.0,0.0,0.0], [0.0, 0.0, 0.0]], [0.0, 0.0], [spec1, spec2], C, false)
 
    return energy(IP, at) - energy(IP.components[1], at)
 end


function plot_dimer(IP, m)
    V2 = IP.components[2]
    elements = collect(chemical_symbol.(V2.basis.zlist.list.data))

    p1 = plot()
    R = [r for r in  0.1:0.01:7.0]
    for el1 in elements
       for el2 in elements
            plot!(p1, R, [HAL.UTILS.dimer_energy(IP, r, AtomicNumber(el1), AtomicNumber(el2)) for r in R], label="$(el1), $(el2)")
       end
    end

    ylims!(-3.0, 3.0)
    ylabel!("Energy [eV/atom]")
    xlabel!("r [Å]")

    savefig("dimers_$(m).pdf")
end


function get_coeff(al, B, ncomms, weights, Vref; sparsify=false, return_S=false)
    ARDRegression = pyimport("sklearn.linear_model")["ARDRegression"]
    BRR = pyimport("sklearn.linear_model")["BayesianRidge"]

    dB = LsqDB("", B, al)
    Ψ, Y = IPFitting.Lsq.get_lsq_system(dB, verbose=true,
                                Vref=Vref, Ibasis = :,Itrain = :,
                                weights=weights, regularisers = [])

    if sparsify
        clf = ARDRegression()
        clf.fit(Ψ, Y)
        norm(clf.coef_)
        inds = findall(clf.coef_ .!= 0)

        clf = BRR()#fit_intercept=false)
        clf.fit(Ψ[:,inds], Y)

        S_inv = clf.alpha_ * Diagonal(ones(length(inds))) + clf.lambda_ * Symmetric(transpose(Ψ[:,inds])* Ψ[:,inds])
        S = Symmetric(inv(S_inv))
        m = clf.lambda_ * (Symmetric(S)*transpose(Ψ[:,inds])) * Y

        for e in reverse([10.0^-i for i in 1:50])
           try
               global d = MvNormal(m, Symmetric(S) - (minimum(eigvals(Symmetric(S))) - e)*I)
               break
           catch
           end
        end
        
        #d = MvNormal(m, Symmetric(S))
        c_samples = rand(d, ncomms);

        c = zeros(length(B))
        c[inds] = clf.coef_
        
        k = zeros(length(B), ncomms)
        for i in 1:ncomms
            _k = zeros(length(B))
            _k[inds] = c_samples[:,i]
            k[:,i] = _k
        end

        if return_S
            return c, c_samples, S
        else
            return c, c_samples
        end
    else
        clf = BRR()#fit_intercept=false)
        clf.fit(Ψ, Y)

        S_inv = clf.alpha_ * Diagonal(ones(length(Ψ[1,:]))) + clf.lambda_ * Symmetric(transpose(Ψ)* Ψ)
        S = Symmetric(inv(S_inv))
        m = clf.lambda_ * (Symmetric(S)*transpose(Ψ)) * Y

        for e in reverse([10.0^-i for i in 1:50])
            try
                global d = MvNormal(m, Symmetric(S) - (minimum(eigvals(Symmetric(S))) - e)*I)
                break
            catch
            end
        end
         
        #d = MvNormal(m, Symmetric(S))
        c_samples = rand(d, ncomms);
        c = clf.coef_

        if return_S
            return c, c_samples, S
        else
            return c, c_samples
        end
    end
end

function get_E_uncertainties(al_test, B, Vref, c, k)
    nIPs = length(k[1,:])
    nconfs = length(al_test)

    #IP = SumIP(Vref, JuLIP.MLIPs.combine(B, c))
    #IPs = [SumIP(Vref, JuLIP.MLIPs.combine(B, k[:,i])) for i in 1:nIPs]

    Pl = zeros(nconfs)
    El = zeros(nconfs)
    Cl = Vector(undef, nconfs)
    Threads.@threads for i in 1:nconfs
        at = al_test[i]
        nats = length(at.at)

        E = energy(B, at.at)
        E_shift = energy(Vref, at.at)

        Es = [(E_shift + sum(k[:,i] .* E))/nats for i in 1:nIPs];
        #Fs = [sum(k[:,i] .* F) for i in 1:nIPs];
        meanE = (E_shift + sum(c .* E))/nats

        p = sqrt(sum([ (Es[m] - meanE)^2 for m in 1:nIPs])/nIPs)
        e = abs.(meanE .- (at.D["E"][1]/nats))
        cg = configtype(at)

        if p != (Inf, NaN) && e != (Inf, NaN)
            Pl[i] = p
            El[i] = e
            Cl[i] = cg
        end
    end
    inds = findall(0.0 .!= Pl)
    El = El[inds]
    Pl = Pl[inds]
    Cl = Cl[inds]
    inds2 = findall(0.0 .!= El)
    return El[inds2], Pl[inds2], Cl[inds2]
end

function get_E_uncertainties_sites(al_test, B, Vref, c, k)
    nIPs = length(k[1,:])
    nconfs = length(al_test)

    IP = SumIP(Vref, JuLIP.MLIPs.combine(B, c))
    IPs = [SumIP(Vref, JuLIP.MLIPs.combine(B, k[:,i])) for i in 1:nIPs]

    Pl = zeros(nconfs)
    El = zeros(nconfs)
    Cl = Vector(undef, nconfs)
    Threads.@threads for i in 1:nconfs
        at = al_test[i]
        
        mean_site_Es, Es = _get_sites(IPs, at)

        p = sqrt(sum([ (Es[m] .- meanE_site_Es)^2 for m in 1:nIPs])/nIPs)
        e = abs.(mean_site_Es .- (at.D["E"][1]/nats))
        cg = configtype(at)

        if p != (Inf, NaN) && e != (Inf, NaN)
            Pl[i] = p
            El[i] = e
            Cl[i] = cg
        end
    end
    inds = findall(0.0 .!= Pl)
    El = El[inds]
    Pl = Pl[inds]
    Cl = Cl[inds]
    inds2 = findall(0.0 .!= El)
    return El[inds2], Pl[inds2], Cl[inds2]
end

function get_F_uncertainties(al_test, B, Vref, c, k)
    IP = SumIP(Vref, JuLIP.MLIPs.combine(B, c))

    nIPs = length(k[1,:])
    nconfs = length(al_test)

    Pl = zeros(nconfs)
    Fl = zeros(nconfs)
    Cl = Vector(undef, nconfs)
    Threads.@threads for i in 1:nconfs
        at = al_test[i]
        nats = length(at.at)

        E = energy(B, at.at)
        F = forces(B, at.at)

        E_shift = energy(Vref, at.at)

        Es = [(E_shift + sum(k[:,i] .* E))/nats for i in 1:nIPs];
        Fs = [sum(k[:,i] .* F) for i in 1:nIPs];

        meanE = (E_shift + sum(c .* E))/nats
        #varE = sum([ (Es[i] - meanE)^2 for i in 1:nIPs])/nIPs

        #meanE = mean(Es)
        #meanF = mean(Fs)
        meanF = sum(c .* F)
        meanE = (E_shift + sum(c .* E))/nats
        #varF =  sum([ 2*(Es[i] - meanE)*(Fs[i] - meanF) for i in 1:nIPs])/nIPs
        
        #stdF = sqrt(sum(vcat([vcat(Fs[m]...) .- vcat(meanF...) for m in 1:nIPs]...).^2)/length(nIPs))
        varF =  sum([ 2*(Es[i] - meanE)*(Fs[i] - meanF) for i in 1:nIPs])/nIPs #2*(Es[i] - meanE)*

        #F = (norm.(varF))
        #p = sqrt(mean(vcat(varF...).^2))
        #f = maximum(vcat(F...) .- at.D["F"])= forces(IP, at.at)
        #p 
        #F = forces(IP, at.at)

        try
            #p = maximum(norm.(varF))
            p = maximum(norm.(varF)) #varF
            f = sqrt(mean((vcat(meanF...) .- at.D["F"]).^2))

            Pl[i] = p
            Fl[i] = f
            Cl[i] = configtype(at)
        catch
        end
    end
    inds = findall(0.0 .!= Pl)
    return Fl[inds], Pl[inds], Cl[inds]
end

function _get_sites(IPs, at)
    nIPs = length(IPs)
    nats = length(at)
	Es = zeros(nIPs, nats)

	for j in 1:nIPs
		Es[j,:] = [sum([ site_energy(V, at, i0) for V in IPs[j].components[2:end]]) for i0 in 1:nats]
	end

	mean_site_Es = [mean(Es[:,n]) for n in 1:nats]

	return mean_site_Es, Es
end

function get_F_uncertainties_sites(al_test, B, Vref, c, k)
    nIPs = length(k[1,:])
    nconfs = length(al_test)

    IP = SumIP(Vref, JuLIP.MLIPs.combine(B, c))
    IPs = [SumIP(Vref, JuLIP.MLIPs.combine(B, k[:,i])) for i in 1:nIPs]

    Pl = zeros(nconfs)
    Fl = zeros(nconfs)
    Cl = Vector(undef, nconfs)
    #Esl = Vector(undef, nconfs)
    Threads.@threads for i in 1:nconfs
        at = al_test[i]
        #nats = length(at.at)

        ### ENERGY
        mean_site_Es, Es = _get_sites(IPs, at.at)

        ### FORCES
        F = forces(B, at.at)
        Fs = [sum(k[:,l] .* F) for l in 1:nIPs];
        meanF = mean(Fs)

        ### UNCERTAINTY FORCE

        varF = sum([ 2*(Es[m,:] .- mean_site_Es) .* (Fs[m] - meanF) for m in 1:nIPs])/nIPs

        # E = energy(B, at.at)
        # F = forces(B, at.at)

        # E_shift = energy(Vref, at.at)

        # Es = [E_shift + sum(k[:,i] .* E) for i in 1:nIPs];
        # Fs = [sum(k[:,i] .* F) for i in 1:nIPs];

        # meanE = mean(Es)
        # #varE = sum([ (Es[i] - meanE)^2 for i in 1:nIPs])/nIPs

        # meanF = mean(Fs)
        # varF =  sum([ 2*((Es[i] - meanE)/nats)*(Fs[i] - meanF) for i in 1:nIPs])/nIPs
        #varF =  sum([ (Fs[i] - meanF) for i in 1:n])/n #2*(Es[i] - meanE)*

        cg = configtype(at)
        F = forces(IP, at.at)
        #p = (norm.(varF) ./ 0.2 + norm.(F))
        #p = sqrt(mean(vcat(varF...).^2))
        #f = maximum(vcat(F...) .- at.D["F"])
        p = maximum(norm.(varF))
        f = sqrt(mean((vcat(F...) .- at.D["F"]).^2))
        if p != (Inf, NaN) && f != (Inf, NaN)
            Pl[i] = p
            Fl[i] = f
            Cl[i] = cg
        end
    end
    inds = findall(0.0 .!= Pl)
    Fl = Fl[inds]
    Pl = Pl[inds]
    Cl = Cl[inds]
    inds2 = findall(0.0 .!= Fl)
    return Fl[inds2], Pl[inds2], Cl[inds2]
end

function HAL_E_dev(al, al_test, B, ncomms, iters, nadd, weights, Vref, plot_dict; sites=true, sparsify=true)
    for i in 1:iters
        @info("ITERATION $(i)")
        c, k = get_coeff(al, B, ncomms, weights, Vref, sparsify)

        IP = SumIP(Vref, JuLIP.MLIPs.combine(B, c))

        save_dict("./IP_HAL_$(i).json", Dict("IP" => write_dict(IP)))

        @info("HAL ERRORS OF ITERATION $(i)")
        add_fits!(IP, al, fitkey="IP2")
        rmse_, rmserel_ = rmse(al; fitkey="IP2");
        rmse_table(rmse_, rmserel_)

        if sites
            El_train, Pl_train, Cl_train = get_E_uncertainties_sites(al, B, Vref, c, k)
            El_test, Pl_test, Cl_test = get_E_uncertainties_sites(al_test, B, Vref, c, k)
        else
            El_train, Pl_train, Cl_train = get_E_uncertainties(al, B, Vref, c, k)
            El_test, Pl_test, Cl_test = get_E_uncertainties(al_test, B, Vref, c, k)
        end

        train_shapes = [plot_dict[config_type] for config_type in Cl_train]
        test_shapes = [plot_dict[config_type] for config_type in Cl_test]

        p = plot()
        scatter!(p, Pl_test .+1E-10, abs.(El_test) .+1E-10, markershapes=test_shapes, legend=:bottomright, label="test")
        scatter!(p, Pl_train .+1E-10, abs.(El_train) .+1E-10, markershapes=train_shapes, label="train")
        xlabel!(p, L" \sigma^2(x)")
        ylabel!(p, L" \Delta E  \quad [eV/atom]")
        #hline!([0.001], color="black", label="1 meV")
        display(p)
        savefig("HAL_E_$(i).png")

        # @show("USING VASP")
        # converted_configs = []
        # for (j, selected_config) in enumerate(al_test[inds])
        #     #try
        #     #at, py_at = 
        #     HAL.CALC.VASP_calculator(selected_config.at, "HAL_$(i)", i, j, calc_settings)
        #     #catch
        #     #    @show("VASP failed?")
        #     #end
        #     #al = vcat(al, at)
        #     #push!(converted_configs, at)
        # end

        #save_configs(converted_configs, i)

        Pl_test_fl = filter(!isnan, Pl_test)
        maxvals = sort(Pl_test_fl)[end-nadd:end]

        inds = [findall(Pl_test .== maxvals[end-i])[1] for i in 0:nadd]
        not_inds = filter!(x -> x ∉ inds, collect(1:length(al_test)))

        al = vcat(al, al_test[inds])

        save_configs(al, i, "TRAIN")
        save_configs(al_test[not_inds], i, "TEST")
    end
    return al
end

function HAL_E(al, al_test, B, ncomms, iters, nadd, weights, Vref; sparsify=true)
    for i in 1:iters
        @info("ITERATION $(i)")
        c, k = get_coeff(al, B, ncomms, weights, Vref, sparsify)

        IP = SumIP(Vref, JuLIP.MLIPs.combine(B, c))

        @info("HAL ERRORS OF ITERATION $(i)")
        add_fits_serial!(IP, al, fitkey="IP2")
        rmse_, rmserel_ = rmse(al; fitkey="IP2");
        rmse_table(rmse_, rmserel_)

        El_train, Pl_train = get_E_uncertainties(al, B, Vref, c, k)
        El_test, Pl_test = get_E_uncertainties(al_test, B, Vref, c, k)
        scatter(Pl_test, El_test, yscale=:log10, xscale=:log10, legend=:bottomright, label="test")
        scatter!(Pl_train, El_train, yscale=:log10, xscale=:log10,label="train")
        xlabel!(L" \sigma^2(x)")
        ylabel!(L" \Delta E  \quad [eV/atom]")
        hline!([0.001], color="black", label="1 meV")
        savefig("HAL_E_$(i).png")

        Pl_test_fl = filter(!isnan, Pl_test)
        maxvals = sort(Pl_test_fl)[end-nadd:end]

        inds = [findall(Pl_test .== maxvals[end-i])[1] for i in 0:nadd]

        # @show("USING VASP")
        # converted_configs = []
        # for (j, selected_config) in enumerate(al_test[inds])
        #     #try
        #     #at, py_at = 
        #     HAL.CALC.VASP_calculator(selected_config.at, "HAL_$(i)", i, j, calc_settings)
        #     #catch
        #     #    @show("VASP failed?")
        #     #end
        #     #al = vcat(al, at)
        #     #push!(converted_configs, at)
        # end

        #save_configs(converted_configs, i)

        al = vcat(al, al_test[inds])
        #save_configs(al_test[inds], i)
        save_configs(al, i)
    end
end

function HAL_F(al, al_test, B, ncomms, iters, nadd, weights, Vref, plot_dict; sites=true, sparsify=true)
    for i in 1:iters
        all_configtypes = unique(configtype.(al))
        nats = length(al)

        c, k = get_coeff(al, B, ncomms, weights, Vref, sparsify)

        IP = SumIP(Vref, JuLIP.MLIPs.combine(B, c))

        save_dict("./IP_HAL_$(i).json", Dict("IP" => write_dict(IP)))

        #add_fits!(IP, al, fitkey="IP2")
        #rmse_, rmserel_ = rmse(al; fitkey="IP2");
        #rmse_table(rmse_, rmserel_)

        if sites
            Fl_train, Pl_train, Cl_train = get_F_uncertainties_sites(al, B, Vref, c, k)
            Fl_test, Pl_test, Cl_test = get_F_uncertainties_sites(al_test, B, Vref, c, k)
        else
            Fl_train, Pl_train, Cl_train = get_F_uncertainties(al, B, Vref, c, k)
            Fl_test, Pl_test, Cl_test = get_F_uncertainties(al_test, B, Vref, c, k)
        end

        train_shapes = [plot_dict[config_type] for config_type in Cl_train]
        test_shapes = [plot_dict[config_type] for config_type in Cl_test]

        envs_train = sum([length(at.at) for at in al])
        envs_test = sum([length(at.at) for at in al_test])

        perc = (round((envs_train/(envs_train + envs_test)), digits=2)*100)

        p = plot()
        title!(p, "$(perc)% of DB")
        scatter!(p, Pl_test, Fl_test, markershapes=test_shapes, yscale=:log10, xscale=:log10, legend=:bottomright, label="test")
        scatter!(p, Pl_train, Fl_train, markershapes=train_shapes, yscale=:log10, xscale=:log10,label="train")
        xlabel!(p,L"\sigma_{F} \quad \textrm{[eV/Å]}")
	    ylabel!(p, "RMSE Force Error [eV/Å]")
        hline!(p,[0.075], color="black", label="ACE npj", linestyle=:dash)
        hline!(p,[mean(Fl_test)], color=1, label="test", linestyle=:dash)
        hline!(p,[mean(Fl_train)], color=2, label="train", linestyle=:dash)
        #display(p)
        savefig("HAL_F_$(i).pdf")

        Pl_test_fl = filter(!isnan, Pl_test)
        maxvals = sort(Pl_test_fl)[end-nadd:end]

        inds = [findall(Pl_test .== maxvals[end-i])[1] for i in 0:nadd]
        not_inds = filter!(x -> x ∉ inds, collect(1:length(al_test)))

        al = vcat(al, al_test[inds])

        save_configs(al, i, "TRAIN")
        save_configs(al_test[not_inds], i, "TEST")
    end
    return al
end

function HAL_F2(al, al_test, B, ncomms, iters, nadd, weights, Vref, plot_dict; sites=true, sparsify=true)
    for i in 1:iters
        c, k = get_coeff(al, B, ncomms, weights, Vref, sparsify)

        IP = SumIP(Vref, JuLIP.MLIPs.combine(B, c))

        save_dict("./IP_HAL_$(i).json", Dict("IP" => write_dict(IP)))

        #add_fits!(IP, al, fitkey="IP2")
        #rmse_, rmserel_ = rmse(al; fitkey="IP2");
        #rmse_table(rmse_, rmserel_)

        if sites
            Fl_train, Pl_train, Cl_train = get_F_uncertainties_sites(al, B, Vref, c, k)
            Fl_test, Pl_test, Cl_test = get_F_uncertainties_sites(al_test, B, Vref, c, k)
        else
            Fl_train, Pl_train, Cl_train = get_F_uncertainties(al, B, Vref, c, k)
            Fl_test, Pl_test, Cl_test = get_F_uncertainties(al_test, B, Vref, c, k)
        end

        #train_shapes = [plot_dict[config_type] for config_type in Cl_train]
        #test_shapes = [plot_dict[config_type] for config_type in Cl_test]

        #envs_train = sum([length(at.at) for at in al])
        #envs_test = sum([length(at.at) for at in al_test])

        #perc = (round((envs_train/(envs_train + envs_test)), digits=2)*100)

        config_types = unique(vcat(Cl_train, Cl_test))

        @show length(config_types)

        p = plot(layout=length(config_types), size=(1000,1000))
        #title!(p, "$(perc)% of DB")
        @show config_types

        for (i,cfg_type) in enumerate(config_types)
            inds_train = findall(Cl_train .== cfg_type)
            inds_test = findall(Cl_test .== cfg_type)
            scatter!(p, Pl_test[inds_test], Fl_test[inds_test], subplot=i, legend=:bottomright, label="")
            scatter!(p, Pl_train[inds_train], Fl_train[inds_train], subplot=i, label="$(cfg_type)")
        end
        # xlabel!(p,L"\sigma_{F} \quad \textrm{[eV/Å]}")
	    # ylabel!(p, "RMSE Force Error [eV/Å]")
        # hline!(p,[0.075], color="black", label="ACE npj", linestyle=:dash)
        # hline!(p,[mean(Fl_test)], color=1, label="test", linestyle=:dash)
        # hline!(p,[mean(Fl_train)], color=2, label="train", linestyle=:dash)
        #display(p)
        savefig("HAL_F_$(i).pdf")

        Pl_test_fl = filter(!isnan, Pl_test)
        maxvals = sort(Pl_test_fl)[end-nadd:end]

        inds = [findall(Pl_test .== maxvals[end-i])[1] for i in 0:nadd]
        not_inds = filter!(x -> x ∉ inds, collect(1:length(al_test)))

        al = vcat(al, al_test[inds])

        save_configs(al, i, "TRAIN")
        save_configs(al_test[not_inds], i, "TEST")
    end
    return al
end


function save_configs(al, i, fname; energy_key="energy", force_key="forces", virial_key="virial")
    py_write = pyimport("ase.io")["write"]
    al_save = []
    for at in al
        py_at = ASEAtoms(at.at)

        D_info = PyDict(py_at.po[:info])
        D_arrays = PyDict(py_at.po[:arrays])

        D_info["config_type"] = configtype(at) #"HAL_$(i)_" * configtype(at)
        try D_info[energy_key] = at.D["E"] catch end
        try D_info[virial_key] = [at.D["V"][1], at.D["V"][6], at.D["V"][5], at.D["V"][6], at.D["V"][2], at.D["V"][4], at.D["V"][5], at.D["V"][4], at.D["V"][3]] catch end
        try D_arrays[force_key] = transpose(reshape(at.D["F"], 3, length(at.at))) catch end

        py_at.po[:info] = D_info
        py_at.po[:arrays] = D_arrays

        push!(al_save, py_at.po)
    end
    py_write("HAL_$(fname)_$(i).xyz", PyVector(al_save))
end

end
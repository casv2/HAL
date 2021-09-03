module RUN

using IPFitting
using JuLIP
using ASE
using HMD
using LinearAlgebra
using Plots
using ACE
using JuLIP.MLIPs: SumIP
using IPFitting: Dat
using Distributions

function do_fit(B, Vref, al, weights, ncoms; reweight=false)#; calc_err=true)
    dB = IPFitting.Lsq.LsqDB("", B, al);

    if reweight
        for (m,at) in enumerate(al)
            meanF = mean(abs.(vcat(at.D["F"]...)))
            config = "HMD_" * string(m)
            al[m].configtype = config

            weights[config] = Dict("E" => 15.0, "F" => 1/meanF, "V" => 1.0)
        end
    end   

    @show weights

    Ψ, Y = IPFitting.Lsq.get_lsq_system(dB, verbose=true,
                                Vref=Vref, Ibasis = :,Itrain = :,
                                weights=weights, regularisers = [])

    α, β, c, lml_score = HMD.BRR.maxim_hyper(Ψ, Y)

    #@show lml_score
    
    k = HMD.BRR.do_brr(Ψ, Y, α, β, ncoms);

    #c, c_samples = HMD.BRR.get_coeff(Ψ, Y, ncoms)
    
    IP = JuLIP.MLIPs.SumIP(Vref, JuLIP.MLIPs.combine(B, c))
    
    #if calc_err 
    add_fits!(IP, al, fitkey="IP")
    rmse_, rmserel_ = rmse(al; fitkey="IP");
    rmse_table(rmse_, rmserel_)
    #end
    
    return IP, k
end

function run_HMD(Vref, weights, al, start_configs, run_info, calc_settings, Binfo)#, nsteps=10000)
    # if refit == false
    #     IP, c_samples = do_fit(B, Vref, al, weights, run_info["ncoms"])
    # end
    for (j,start_config) in enumerate(start_configs)
        config_type = configtype(start_config)
        for l in 1:convert(Int,run_info["HMD_iters"])
            init_config = deepcopy(start_config)
            #init_config = deepcopy(al[end])
            m = (j-1)*run_info["HMD_iters"] + l

            # if run_info["optim_basis"] == true
            #     maxN, maxdeg = HMD.OPTIM.find_N_deg(Binfo, Vref, weights, al)
            #     Binfo["N"] = maxN
            #     Binfo["deg"] = maxdeg
            #     @info("FOUND OPTIMUM BASIS: N=$(maxN), D=$(maxdeg)")
            # end
            
            #R = minimum(IPFitting.Aux.rdf(al, 4.0)) + run_info["Rshift"]

            Bsite = rpi_basis(species = Binfo["Z"],
                N = Binfo["N"],       # correlation order = body-order - 1
                maxdeg = Binfo["deg"],  # polynomial degree
                r0 = Binfo["r0"],     # estimate for NN distance
                rin = Binfo["R"], rcut = Binfo["Nrcut"],   # domain for radial basis (cf documentation)
                pin = 2) 

            Bpair = pair_basis(species = Binfo["Z"],
                r0 = Binfo["r0"],
                maxdeg = Binfo["2B"],
                rcut = Binfo["2Brcut"],
                pcut = 1,
                pin = 0) 

            B = JuLIP.MLIPs.IPSuperBasis([Bpair, Bsite]);

            if haskey(run_info, "refit")
                if m % run_info["refit"] == 1
                    global IP, k = do_fit(B, Vref, al, weights, run_info["ncoms"])
                end
            else
                IP, k = do_fit(B, Vref, al, weights, run_info["ncoms"])
            end

            if config_type ∉ keys(run_info)
                D = copy(run_info["default"])
                run_info[config_type] = D
            end

            E_tot, E_pot, E_kin, T, P, varEs, varFs, selected_config = run(IP,Vref, B, k, 
                    init_config.at, 
                    nsteps=run_info["nsteps"], 
                    temp=run_info[config_type]["temp"], 
                    dt=run_info[config_type]["dt"], 
                    τstep=run_info[config_type]["τstep"], 
                    dτ=run_info[config_type]["dτ"], 
                    maxp=run_info[config_type]["maxp"],
                    γ=run_info[config_type]["γ"],
                    swap=run_info[config_type]["swap"],
                    vol=run_info[config_type]["vol"],
                    heat=run_info[config_type]["heat"],
                    minR=run_info[config_type]["minR"])
            
            plot_HMD(E_tot, E_pot, E_kin, T, P, m, k=1)

            try 
                if calc_settings["calculator"] == "DFTB"
                    at, py_at = HMD.CALC.DFTB_calculator(selected_config, config_type, calc_settings)
                elseif calc_settings["calculator"] == "ORCA"
                    at, py_at = HMD.CALC.ORCA_calculator(selected_config, config_type, calc_settings)
                elseif calc_settings["calculator"] == "CASTEP"
                    at, py_at = HMD.CALC.CASTEP_calculator(selected_config, config_type, calc_settings)
                elseif calc_settings["calculator"] == "NRLTB"
                    at, py_at = HMD.CALC.NRLTB_calculator(selected_config, config_type, m)
                elseif calc_settings["calculator"] == "NRLTBpy3"
                    at, py_at = HMD.CALC.NRLTBpy3_calculator(selected_config, config_type, calc_settings, m)
                end

                al = vcat(al, at)
                write_xyz("./HMD_it$(m).xyz", py_at)
            catch
                println("Iteration failed! Calulator probably failed!")
            end
        end
    end
    return al
end

function energy_uncertainty(IP, IPs, at)
	nIPs = length(IPs)

	Es = [energy(IP, at) for IP in IPs]
	meanE = energy(IP, at)

	stdE = sqrt(sum([ (Es[i] - meanE).^2 for i in 1:nIPs])/nIPs)/length(at)
	return stdE, meanE
end

function swap_step(at)
	ind1, ind2 = rand(1:length(at), 2)

	M1 = at.M[ind1]
	Z1 = at.Z[ind1]

	M2 = at.M[ind2]
	Z2 = at.Z[ind2]

	at.M[ind1] = M2
	at.M[ind2] = M1

	at.Z[ind1] = Z2
	at.Z[ind2] = Z1

	return at
end

function vol_step(at)
    d = Normal()
	scale = 1 + (rand(d) * 0.01)
    C1 = at.cell
    C2 = at.cell*scale .+ (rand(d, (3,3)) * 0.01)
    s = C2 / C1
    for i in 1:length(at)
        at.X[i] = at.X[i]' * s
    end
    at = set_cell!(at, C2)
    at = set_positions!(at, at.X * scale)
	return at
end

function _get_site(IP, at)
    nats = length(at)
    Es = [sum([site_energy(V, at, i0) for V in IP.components[2:end]]) for i0 in 1:nats]
    return Es
end

function get_site_uncertainty(IP, IPs, at)
    # nIPs = length(IPs)
    # Es = zeros(length(at), nIPs)

    # Threads.@threads for j in 1:nIPs
    #         Es[:,j] = _get_site(IPs[j], at)
    # end

    # oneB = energy(IP.components[1], at)
    # mean_E = _get_site(IP, at)

    # E_diff =  std(Es .- mean_E, dims=2)

    # @show Es
    # @show E_diff

    #E = energy(IP, at)/length(at)
    #Es_rmse = sqrt(mean([(energy(IP, at)/length(at) - E).^2 for IP in IPs]));
    
    #F = forces(IP, at)
    #Fs_rmse = sqrt(mean(reduce(vcat, [vcat((forces(IP, at) - F)...).^2 for IP in IPs])))
    
    F = forces(IP, at)
    Fs = Vector(undef, length(IPs))

    @Threads.threads for i in 1:nIPs
        Fs[i] = forces(IPs[i], at)
    end

    dFn = norm.(sum([ (Fs[m] - F) for m in 1:length(IPs)])/length(IPs))
    Fn = norm.(F)

    p = mean(dFn ./ (Fn .+ 0.1))

    #V = virial(IP, at)
    #Vs_rmse = sqrt(mean(reduce(vcat, [vcat((virial(IP, at) - V)...).^2 for IP in IPs])))
    
    #p = 15 * Es_rmse + Fs_rmse + Vs_rmse
    #p = Fs_rmse #/ (mean(abs.(vcat(F...))) + 0.1)

    return p, energy(IP, at)
end

function run(IP, Vref, B, k, at; γ=0.02, nsteps=100, temp=0, dt=1.0, τstep=50, dτ=0.01, maxp=0.15, minR=2.0, var=true, A=1e-6, swap=false, vol=false, heat=false)
    E_tot = zeros(nsteps)
    E_pot = zeros(nsteps)
    E_kin = zeros(nsteps)
    T = zeros(nsteps)
    P = zeros(nsteps)
    varEs = zeros(nsteps)
    varFs = zeros(nsteps)

    E0 = energy(IP, at)

    at = HMD.MD.MaxwellBoltzmann_scale(at, temp)
    at = HMD.MD.Stationary(at)

    nIPs = length(k[1,:])
    IPs = [SumIP(Vref, JuLIP.MLIPs.combine(B, k[:,i])) for i in 1:nIPs]

    running = true

    i = 1
    τ = 0
    while running && i < nsteps
        if heat
            #at = HMD.COM.VelocityVerlet_com_Zm(IP, IPs, at, dt * HMD.MD.fs, A; τ = 0.0)
            #at = HMD.COM.VelocityVerlet_com_langevin(IP, IPs, at, dt * HMD.MD.fs, temp * HMD.MD.kB, γ=γ, τ=τ)
        else
            τ = 0
            at = HMD.COM.VelocityVerlet_com(IP, IPs, at, dt * HMD.MD.fs, τ=τ)
        end
        p, meanE = get_site_uncertainty(IP, IPs, at)
        #else
            # at, p = HMD.COM.VelocityVerlet_com_langevin(IP, IPs, at, dt * HMD.MD.fs, temp * HMD.MD.kB, γ=γ, τ=τ)
            #at, p = HMD.COM.VelocityVerlet_com_Zm(IP, IPs, at, dt, A; τ = 0.0)
        #end
        P[i] = p
        Ek = ((0.5 * sum(at.M) * norm(at.P ./ at.M)^2)/length(at.M)) / length(at.M)
        Ep = (energy(IP, at) - E0) / length(at.M)
        E_tot[i] = Ek + Ep
        E_pot[i] = Ep
        E_kin[i] = Ek
        T[i] = Ek / (1.5 * HMD.MD.kB)
        cur_al = Dat[]
        push!(cur_al, Dat(at, "HMD"))
        R = minimum(IPFitting.Aux.rdf(cur_al, 4.0))
        @show p, τ, R
        if i % τstep == 0 && swap
            at = deepcopy(at)
            p_at, E_at = get_site_uncertainty(IP, IPs, at)
            at_new = swap_step(at)
            p_at_new, E_at_new = get_site_uncertainty(IP, IPs, at_new)
            #C = exp( - ((E_at - p_at) - (E_at_new - p_at_new)) / (HMD.MD.kB * temp))
            C = exp( - (E_at - E_at_new) / (HMD.MD.kB * temp))
            @show C
            if rand() < C
                println("SWAP ACCEPTED")
                at = at_new
            end
        end
        if i % (τstep/10) == 0 && vol
            at = deepcopy(at)
            p_at, E_at = get_site_uncertainty(IP, IPs, at)
            at_new = vol_step(at)
            p_at_new, E_at_new = get_site_uncertainty(IP, IPs, at_new)
            C = exp( - (E_at - E_at_new) / (HMD.MD.kB * temp))
            @show C
            if rand() < C
                println("VOL STEP ACCEPTED")
                at = at_new
            end
        end
        if p > maxp || R < minR
            running = false
        end
        if i % τstep == 0
            τ += dτ
        end
        #push!(cfgs, at)
        i+=1
    end

    # if i < nsteps
    #     selected_config = cfgs[end]
    # else
    #     max_ind = findmax(P)[2]
    #     selected_config = cfgs[max_ind]
    # end
    
    return E_tot[1:i], E_pot[1:i], E_kin[1:i], T[1:i], P[1:i], varEs[1:i], varFs[1:i], at #selected_config
end

function plot_HMD(E_tot, E_pot, E_kin, T, P, i; k=50) # varEs,
    p1 = plot()
    plot!(p1,E_tot[1:end-k], label="")
    plot!(p1,E_kin[1:end-k], label="")
    plot!(p1,E_pot[1:end-k], label="")
    ylabel!(p1, "Energy (eV)")
    p2 = plot()
    plot!(p2, T[1:end-k],label="")
    ylabel!(p2, "T (K)")
    p4 = plot()
    plot!(p4, P[1:end-k],label="")
    xlabel!(p4,"MDstep")
    ylabel!(p4, "maximum F_s")
    p5 = plot(p1, p2, p4, size=(400,550), layout=grid(3, 1, heights=[0.6, 0.2, 0.2]))
    savefig("./HMD_$(i).pdf")
end


end

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

function do_fit(B, Vref, al, weights, ncoms)#; calc_err=true)
    dB = IPFitting.Lsq.LsqDB("", B, al);

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

function run_HMD(B, Vref, weights, al, start_configs, run_info, calc_settings)#, nsteps=10000)
    # if refit == false
    #     IP, c_samples = do_fit(B, Vref, al, weights, run_info["ncoms"])
    # end
    for (j,start_config) in enumerate(start_configs)
        config_type = configtype(start_config)
        for l in 1:convert(Int,run_info["HMD_iters"])
            init_config = deepcopy(start_config)
            m = (j-1)*run_info["HMD_iters"] + l

            # if run_info["optim_basis"] == true
            #     maxN, maxdeg = HMD.OPTIM.find_N_deg(Binfo, Vref, weights, al)
            #     Binfo["N"] = maxN
            #     Binfo["deg"] = maxdeg
            #     @info("FOUND OPTIMUM BASIS: N=$(maxN), D=$(maxdeg)")
            # end
            
            #R = minimum(IPFitting.Aux.rdf(al, 4.0))

            # Bsite = rpi_basis(species = Binfo["Z"],
            #     N = Binfo["N"],       # correlation order = body-order - 1
            #     maxdeg = Binfo["deg"],  # polynomial degree
            #     r0 = Binfo["r0"],     # estimate for NN distance
            #     rin = R, rcut = Binfo["Nrcut"],   # domain for radial basis (cf documentation)
            #     pin = 2) 

            # Bpair = pair_basis(species = Binfo["Z"],
            #     r0 = Binfo["r0"],
            #     maxdeg = Binfo["2B"],
            #     rcut = Binfo["2Brcut"],
            #     pcut = 1,
            #     pin = 0) 

            # B = JuLIP.MLIPs.IPSuperBasis([Bpair, Bsite]);

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

            rand_ind = rand(1:length(al))

            init_config = deepcopy(al[rand_ind])

            E_tot, E_pot, E_kin, T, P, varEs, varFs, selected_config = run(IP,Vref, B, k, 
                    init_config.at, 
                    nsteps=run_info["nsteps"], 
                    temp=run_info[config_type]["temp"], 
                    dt=run_info[config_type]["dt"], 
                    τstep=run_info[config_type]["τstep"], 
                    dτ=run_info[config_type]["dτ"], 
                    maxp=run_info[config_type]["maxp"],
                    γ=run_info[config_type]["γ"],
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

function run(IP, Vref, B, k, at; γ=0.02, nsteps=100, temp=0, dt=1.0, τstep=50, dτ=0.01, maxp=0.15, minR=2.0, var=true)
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
        if temp == 0
            at, p = HMD.COM.VelocityVerlet_com(IP, IPs, at, dt * HMD.MD.fs, τ=τ)
        else
            at, p = HMD.COM.VelocityVerlet_com_langevin(IP, IPs, at, dt * HMD.MD.fs, temp * HMD.MD.kB, γ=γ, τ=τ)
        end
        P[i] = p
        Ek = ((0.5 * sum(at.M) * norm(at.P ./ at.M)^2)/length(at.M)) / length(at.M)
        Ep = (energy(IP, at) - E0) / length(at.M)
        E_tot[i] = Ek + Ep
        E_pot[i] = Ep
        E_kin[i] = Ek
        T[i] = Ek / (1.5 * HMD.MD.kB)
        @show p, τ
        al = Dat[]
        push!(al, Dat(at=at, E=energy(IP,at), F=forces(IP,at), V=virial(IP,at)))
        R = minimum(IPFitting.Aux.rdf(al, 4.0))
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

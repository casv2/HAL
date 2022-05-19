module RUN

using IPFitting
using JuLIP
using ASE
using HAL
using LinearAlgebra
using Plots
using JuLIP.MLIPs: SumIP
using IPFitting: Dat
using Distributions
using JSON

function do_fit(B, Vref, al, weights, ncoms; alpha_init=0.1, beta_init=1.0, reweight=false, brrtol=1e-3)#; calc_err=true)
    dB = IPFitting.Lsq.LsqDB("", B, al);

    if reweight
        for (m,at) in enumerate(al)
            meanF = mean(abs.(vcat(at.D["F"]...)))
            config = "HAL_" * string(m)
            al[m].configtype = config

            weights[config] = Dict("E" => 15.0, "F" => 1/meanF, "V" => 1.0)
        end
    end   

    Ψ, Y = IPFitting.Lsq.get_lsq_system(dB, verbose=true,
                                Vref=Vref, Ibasis = :,Itrain = :,
                                weights=weights, regularisers = [])

    @show alpha_init, beta_init

    α, β, c, lml_score = HAL.BRR.maxim_hyper(Ψ, Y, alpha_init, beta_init; brrtol=brrtol)
    
    k = HAL.BRR.do_brr(Ψ, Y, α, β, ncoms);

    #c, c_samples = HAL.BRR.get_coeff(Ψ, Y, ncoms)
    
    IP = JuLIP.MLIPs.SumIP(Vref, JuLIP.MLIPs.combine(B, c))
    
    #if calc_err 
    add_fits!(IP, al, fitkey="IP")
    rmse_, rmserel_ = rmse(al; fitkey="IP");
    rmse_table(rmse_, rmserel_)
    #end
    
    return IP, k, α, β
end

function run_HAL(Vref, weights, al, start_configs, run_info, calc_settings, B)#, nsteps=10000)
    # if refit == false
    #     IP, c_samples = do_fit(B, Vref, al, weights, run_info["ncoms"])
    # end
    global α, β = 1.0,1.0

    for (j,start_config) in enumerate(start_configs)
        config_type = configtype(start_config)
        for l in 1:convert(Int,run_info["HAL_iters"])
            init_config = deepcopy(start_config)
            #init_config = deepcopy(al[end])
            m = (j-1)*run_info["HAL_iters"] + l

            # if run_info["optim_basis"] == true
            #     maxN, maxdeg = HAL.OPTIM.find_N_deg(Binfo, Vref, weights, al)
            #     Binfo["N"] = maxN
            #     Binfo["deg"] = maxdeg
            #     @info("FOUND OPTIMUM BASIS: N=$(maxN), D=$(maxdeg)")
            # end
            
            #R = minimum(IPFitting.Aux.rdf(al, 4.0)) + run_info["Rshift"]

            # Bsite = rpi_basis(species = Binfo["Z"],
            #     N = Binfo["N"],       # correlation order = body-order - 1
            #     maxdeg = Binfo["deg"],  # polynomial degree
            #     r0 = Binfo["r0"],     # estimate for NN distance
            #     rin = Binfo["R"], rcut = Binfo["Nrcut"],   # domain for radial basis (cf documentation)
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
                    global IP, k, α, β = do_fit(B, Vref, al, weights, alpha_init=α, beta_init=β,  run_info["ncoms"])
                end
            else
                IP, k, α, β = do_fit(B, Vref, al, weights, alpha_init=α, beta_init=β, run_info["ncoms"], brrtol=run_info["brrtol"])
            end

            if config_type ∉ keys(run_info)
                D = copy(run_info["default"])
                run_info[config_type] = D
            end

            E_tot, E_pot, E_kin, T, P, Pr, varEs, varFs, selected_config = run(IP,Vref, B, k, 
                    init_config.at, 
                    nsteps=run_info["nsteps"], 
                    temp=run_info[config_type]["temp"], 
                    dt=run_info[config_type]["dt"], 
                    #τstep=run_info[config_type]["τstep"], 
                    #dτ=run_info[config_type]["dτ"], 
                    rτ=run_info[config_type]["rtau"],
                    maxp=run_info[config_type]["maxp"],
                    γ=run_info[config_type]["gamma"],
                    volstep=run_info[config_type]["volstep"],
                    swapstep=run_info[config_type]["swapstep"],
                    swap=run_info[config_type]["swap"],
                    vol=run_info[config_type]["vol"],
                    baro_thermo=run_info[config_type]["baro_thermo"],
                    minR=run_info[config_type]["minR"],
                    Freg=run_info[config_type]["Freg"],
                    μ=run_info[config_type]["mu"],
                    Pr0=run_info[config_type]["Pr0"])
            
            plot_HAL(E_tot, E_pot, E_kin, T, P, Pr, m, k=1)

            try 
                if calc_settings["calculator"] == "DFTB"
                    at, py_at = HAL.CALC.DFTB_calculator(selected_config, config_type, calc_settings)
                elseif calc_settings["calculator"] == "ORCA"
                    at, py_at = HAL.CALC.ORCA_calculator(selected_config, config_type, calc_settings)
                elseif calc_settings["calculator"] == "CASTEP"
                    at, py_at = HAL.CALC.CASTEP_calculator(selected_config, config_type, calc_settings)
                elseif calc_settings["calculator"] == "NRLTB"
                    at, py_at = HAL.CALC.NRLTB_calculator(selected_config, config_type, m)
                elseif calc_settings["calculator"] == "NRLTBpy3"
                    at, py_at = HAL.CALC.NRLTBpy3_calculator(selected_config, config_type, calc_settings, m)
                elseif calc_settings["calculator"] == "Aims"
                    at, py_at = HAL.CALC.Aims_calculator(selected_config, config_type, calc_settings)
                end

                al = vcat(al, at)
                write_xyz("./HAL_it$(m).extxyz", py_at)
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
	found = false
    while found == false
        global ind1, ind2 = rand(1:length(at), 2)

        Z1 = at.Z[ind1]
        Z2 = at.Z[ind2]

        if Z1 != Z2
            found = true
        end
    end

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
	scale = 1 + (rand(d) * 0.005)
    C1 = at.cell
    C2 = at.cell*scale .+ (rand(d, (3,3)) * 0.005)
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

f_w(fi, fm; A=3.0, B=0.5, f0=3.0) = (A + (B * f0 * log(1 + fi/f0 + fm/f0)))^(-1.0)

function run(IP, Vref, B, k, at; γ=0.02, nsteps=100, temp=300, dt=1.0, rτ=0.5, maxp=0.15, minR=2.0, volstep=10, swapstep=10, μ=5e-6, swap=false, vol=false, baro_thermo=false, Freg=0.5, Pr0=0.1) #
    E_tot = zeros(nsteps)
    E_pot = zeros(nsteps)
    E_kin = zeros(nsteps)
    T = zeros(nsteps)
    P = zeros(nsteps)
    Pr = zeros(nsteps)
    mFs = zeros(nsteps)
    mvarFs = zeros(nsteps)
    varEs = zeros(nsteps)
    varFs = zeros(nsteps)

    E0 = energy(IP, at)

    nIPs = length(k[1,:])
    IPs = [SumIP(Vref, JuLIP.MLIPs.combine(B, k[:,i])) for i in 1:nIPs]

    running = true

    i = 1
    τ = 0.0
    #j = 1
    while running && i < nsteps
        # if i < 100
        #     τ = 0.0
        # end

        #if i > temp_steps[j]
        #    j += 1
        #end
        #temp_step = temp_steps[j]
        #temp = temp_dict[temp_step]

        if baro_thermo
            #at = HAL.COM.VelocityVerlet_com_Zm(IP, IPs, at, dt * HAL.MD.fs, A; τ=τ)
            at, varE, varF, Fs, F = HAL.COM.VelocityVerlet_com_langevin_br(IP, IPs, at, dt * HAL.MD.fs, temp * HAL.MD.kB, γ=γ, τ=τ, μ=μ, Pr0=Pr0)
            mvarFs[i] = mean(norm.(varF))
        else
            #τ = 0
            at, varE, varF, Fs, F = HAL.COM.VelocityVerlet_com_langevin(IP, IPs, at, dt * HAL.MD.fs, temp * HAL.MD.kB, γ=γ, τ=τ)
            mvarFs[i] = mean(norm.(varF))
            #at = HAL.COM.VelocityVerlet_com(IP, IPs, at, dt * HAL.MD.fs, τ=τ)
        end
        #else
            # at, p = HAL.COM.VelocityVerlet_com_langevin(IP, IPs, at, dt * HAL.MD.fs, temp * HAL.MD.kB, γ=γ, τ=τ)
            #at, p = HAL.COM.VelocityVerlet_com_Zm(IP, IPs, at, dt, A; τ = 0.0)
        #end
        mFs[i] = mean(norm.(F))
        p = HAL.COM.get_site_uncertainty(F, Fs; Freg=Freg)

        if i > 100
            τ = (rτ * mean(mFs[i-99:i])) / mean(mvarFs[i-99:i])
        else
            τ = 0.0
        end

        P[i] = p
        Pr[i] = -tr(stress(IP,at)) / (3 * HAL.MD.GPa)
        Ek = ((0.5 * sum(at.M) * norm(at.P ./ at.M)^2)/length(at.M)) / length(at.M)
        Ep = (energy(IP, at) - E0) / length(at.M)
        E_tot[i] = Ek + Ep
        E_pot[i] = Ep
        E_kin[i] = Ek
        T[i] = Ek / (1.5 * HAL.MD.kB)
        cur_al = Dat[]
        push!(cur_al, Dat(at, "HAL"))
        R = minimum(IPFitting.Aux.rdf(cur_al, 4.0))
        @show p, τ, R
        if i % swapstep == 0 && swap
            at = deepcopy(at)
            stdE = sqrt(varE)
            E_at = energy(IP, at) - (τ * stdE)
            at_new = swap_step(at)
            E_at_new = energy(IP, at_new) - (τ * stdE)
            #p_at_new, E_at_new = get_site_uncertainty(IP, IPs, at_new)
            #C = exp( - ((E_at - p_at) - (E_at_new - p_at_new)) / (HAL.MD.kB * temp))
            C = exp( - (E_at - E_at_new) / (HAL.MD.kB * temp))
            @show C
            if rand() < C
                println("SWAP ACCEPTED")
                at = at_new
            end
        end
        if i % volstep == 0 && vol
            at = deepcopy(at)
            stdE = sqrt(varE)
            E_at = energy(IP, at) - (τ * stdE)
            at_new = vol_step(at)
            E_at_new = energy(IP, at_new) - (τ * stdE)
            #p_at_new, E_at_new = get_site_uncertainty(IP, IPs, at_new)
            C = exp( - (E_at - E_at_new) / (HAL.MD.kB * temp))
            @show C
            if rand() < C
                println("VOL STEP ACCEPTED")
                at = at_new
            end
        end
        if p > maxp || R < minR
            running = false
        end
        # if i % τstep == 0
        #     τ += dτ
        # end
        #push!(cfgs, at)
        i+=1
    end

    # if i < nsteps
    #     selected_config = cfgs[end]
    # else
    #     max_ind = findmax(P)[2]
    #     selected_config = cfgs[max_ind]
    # end
    
    return E_tot[1:i], E_pot[1:i], E_kin[1:i], T[1:i], P[1:i], Pr[1:i], varEs[1:i], varFs[1:i], at #selected_config
end

function plot_HAL(E_tot, E_pot, E_kin, T, P, Pr, i; k=50) # varEs,
    p1 = plot()
    plot!(p1,E_tot[1:end-k], label="")
    plot!(p1,E_kin[1:end-k], label="")
    plot!(p1,E_pot[1:end-k], label="")
    ylabel!(p1, "Energy (eV/atom)")
    p2 = plot()
    plot!(p2, T[1:end-k],label="")
    ylabel!(p2, "T (K)")
    p4 = plot()
    plot!(p4, P[1:end-k],label="")
    xlabel!(p4,"MDstep")
    ylabel!(p4, "predicted rel. f err.")
    p5 = plot()
    ylabel!(p5, "Pres [GPa]")
    plot!(p5, Pr[1:end-k], label="")
    p5 = plot(p1, p2, p5, p4, size=(400,550), layout=grid(4, 1, heights=[0.4, 0.2, 0.2, 0.2]))
    savefig("./HAL_$(i).pdf")

    D=Dict()
    D["E_tot"]=E_tot
    D["E_kin"]=E_kin
    D["E_pot"]=E_pot
    D["Pr"]=Pr
    D["P"]=P

    stringdata = JSON.json(D)

    # write the file with the stringdata variable information
    open("data_$(i).json", "w") do f
        write(f, stringdata)
    end
end

end

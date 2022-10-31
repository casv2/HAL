module SAMPLE

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
import ASE.ase_io

import HAL.RUN: plot_HAL, _get_site, f_w, vol_step, swap_step, energy_uncertainty


function hal_sample(IP, IPs, start_configs, run_info, sample_interval)
    for (j,start_config) in enumerate(start_configs)
        config_type = configtype(start_config)
        for l in 1:convert(Int,run_info["HAL_iters"])
            init_config = deepcopy(start_config)
            #init_config = deepcopy(al[end])
            m = (j-1)*run_info["HAL_iters"] + l

            if config_type ∉ keys(run_info)
                D = copy(run_info["default"])
                run_info[config_type] = D
            end

            E_tot, E_pot, E_kin, T, U, P, selected_config, samples = run_with_sampling(IP, IPs,
                    init_config.at, 
                    nsteps=run_info["nsteps"], 
                    sample_interval=sample_interval,
                    temp=run_info[config_type]["temp"], 
                    dt=run_info[config_type]["dt"], 
                    rτ=run_info[config_type]["rtau"],
                    Umax=run_info[config_type]["Umax"],
                    γ=run_info[config_type]["gamma"],
                    volstep=run_info[config_type]["volstep"],
                    swapstep=run_info[config_type]["swapstep"],
                    swap=run_info[config_type]["swap"],
                    vol=run_info[config_type]["vol"],
                    baro=run_info[config_type]["baro"],
                    thermo=run_info[config_type]["thermo"],
                    minR=run_info[config_type]["minR"],
                    Freg=run_info[config_type]["Freg"],
                    μ=run_info[config_type]["mu"],
                    Pr0=run_info[config_type]["Pr0"])
            
            plot_HAL(E_tot, E_pot, E_kin, T, U, P, m)

            py_at_configs = []

            for datobj in samples

                py_at = ASEAtoms(datobj.at)
                D_info = PyDict(py_at.po[:info])
                D_arrays = PyDict(py_at.po[:arrays])
                
                for infokey in ["p", "R"]
                    D_info[infokey] = datobj.info[infokey]
                end

                py_at.po[:info] = D_info
                py_at.po[:arrays] = D_arrays                    

                push!(py_at_configs, py_at.po)
            end

            ase_io.write("./samples_$(m).xyz", py_at_configs);
        end
    end
end



function run_with_sampling(IP, IPs, at; γ=0.02, nsteps=100, sample_interval=nsteps+1, temp=300, dt=1.0, rτ=0.5, Umax=0.15, minR=2.0, volstep=10, swapstep=10, μ=5e-6, swap=false, vol=false, baro=false, thermo=false, Freg=0.5, Pr0=0.1)
    # exactly same as run, but also saves samples stuff

    E_tot = zeros(nsteps)
    E_pot = zeros(nsteps)
    E_kin = zeros(nsteps)
    T = zeros(nsteps)
    U = zeros(nsteps)
    P = zeros(nsteps)
    mFs = zeros(nsteps)
    mvarFs = zeros(nsteps)

    E0 = energy(IP, at)

    running = true

    sampled_configs = Vector{Dat}()

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

        if baro && thermo
            #at = HAL.COM.VelocityVerlet_com_Zm(IP, IPs, at, dt * HAL.MD.fs, A; τ=τ)
            at, varE, varF, Fs, F = HAL.COM.VelocityVerlet_com_langevin_br(IP, IPs, at, dt * HAL.MD.fs, temp * HAL.MD.kB, γ=γ, τ=τ, μ=μ, Pr0=Pr0)
            mvarFs[i] = mean(norm.(varF))
        elseif baro == false && thermo == true
            #τ = 0
            at, varE, varF, Fs, F = HAL.COM.VelocityVerlet_com_langevin(IP, IPs, at, dt * HAL.MD.fs, temp * HAL.MD.kB, γ=γ, τ=τ)
            mvarFs[i] = mean(norm.(varF))
        else baro == false && thermo == false 
            at, varE, varF, Fs, F = HAL.COM.VelocityVerlet_com(IP, IPs, at, dt * HAL.MD.fs, τ=τ)
            mvarFs[i] = mean(norm.(varF))
            #at = HAL.COM.VelocityVerlet_com(IP, IPs, at, dt * HAL.MD.fs, τ=τ)
        end
        #else
            # at, p = HAL.COM.VelocityVerlet_com_langevin(IP, IPs, at, dt * HAL.MD.fs, temp * HAL.MD.kB, γ=γ, τ=τ)
            #at, p = HAL.COM.VelocityVerlet_com_Zm(IP, IPs, at, dt, A; τ = 0.0)
        #end
        mFs[i] = mean(norm.(F))

        if i > 100
            @show τ,rτ * mean(mFs[i-99:i]), (rτ * 10 * tanh(  mean(mFs[i-99:i]) / 10 ))
            #τ = (rτ * mean(mFs[i-99:i])) / mean(mvarFs[i-99:i])
            τ = (rτ * 10 * tanh(  mean(mFs[i-99:i]) / 10 )) / mean(mvarFs[i-99:i])
            #τ = (rτ * tanh( mean(mFs[i-99:i])  )) / mean(mvarFs[i-99:i])
        else
            τ = 0.0
        end

        U[i] = HAL.COM.get_site_uncertainty(F, Fs; Freg=Freg)
        P[i] = (-tr(stress(IP,at)) /3) * HAL.MD.GPa
        Ek = 0.5 * sum( at.M .* ( norm.(at.P ./ at.M) .^ 2 ) ) / length(at.M)
        Ep = (energy(IP, at) - E0) / length(at.M)
        E_tot[i] = Ek + Ep
        E_pot[i] = Ep
        E_kin[i] = Ek
        T[i] = Ek / (1.5 * HAL.MD.kB)
        cur_al = Dat[]
        push!(cur_al, Dat(at, "HAL"))
        R = minimum(IPFitting.Aux.rdf(cur_al, 4.0))
        @show U[i]#, τ, R
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
        if U[i] > Umax || R < minR
            running = false
        end
        if i % sample_interval == 0
            println(at.cell)
            new_dat = Dat(at, "sample")
            new_dat.info["R"] = R 
            new_dat.info["p"] = U[i]
            push!(sampled_configs, deepcopy(new_dat))
        end

        i+=1
    end

    # if i < nsteps
    #     selected_config = cfgs[end]
    # else
    #     max_ind = findmax(P)[2]
    #     selected_config = cfgs[max_ind]
    # end
    
    return E_tot[1:i-1], E_pot[1:i-1], E_kin[1:i-1], T[1:i-1], U[1:i-1], P[1:i-1], at, sampled_configs #selected_config
end

end
module RUN

using IPFitting
using JuLIP
using ASE
using HMD
using LinearAlgebra
using Plots
using ACE

function do_fit(B, Vref, al, weights, ncoms)#; calc_err=true)
    dB = IPFitting.Lsq.LsqDB("", B, al);

    Ψ, Y = IPFitting.Lsq.get_lsq_system(dB, verbose=true,
                                Vref=Vref, Ibasis = :,Itrain = :,
                                weights=weights, regularisers = [])

    α, β = HMD.BRR.maxim_hyper(Ψ, Y)
    
    c_samples = HMD.BRR.do_brr(Ψ, Y, α, β, ncoms);
    
    IP = JuLIP.MLIPs.SumIP(Vref, JuLIP.MLIPs.combine(B, c_samples[:,1]))
    
    #if calc_err 
    add_fits_serial!(IP, al, fitkey="IP")
    rmse_, rmserel_ = rmse(al; fitkey="IP");
    rmse_table(rmse_, rmserel_)
    #end
    
    return IP, c_samples
end

function run_HMD(Binfo, Vref, weights, al, start_configs, run_info, calc_settings)#, nsteps=10000)
    for (j,start_config) in enumerate(start_configs)
        config_type = configtype(start_config)
        for l in 1:convert(Int,run_info["HMD_iters"])
            init_config = deepcopy(start_config)
            m = (j-1)*run_info["HMD_iters"] + l

            R = minimum(IPFitting.Aux.rdf(al, 4.0))

            Bsite = rpi_basis(species = Binfo["Z"],
                N = Binfo["N"],       # correlation order = body-order - 1
                maxdeg = Binfo["deg"],  # polynomial degree
                r0 = Binfo["r0"],     # estimate for NN distance
                rin = R, rcut = Binfo["Nrcut"],   # domain for radial basis (cf documentation)
                pin = 2) 

            Bpair = pair_basis(species = Binfo["Z"],
                r0 = Binfo["r0"],
                maxdeg = Binfo["2B"],
                rcut = Binfo["2Brcut"],
                pcut = 1,
                pin = 0) 

            B = JuLIP.MLIPs.IPSuperBasis([Bpair, Bsite]);

            IP, c_samples = do_fit(B, Vref, al, weights, run_info["ncoms"])

            E_tot, E_pot, E_kin, T, P, varEs, varFs, cfgs = run(IP, B, Vref, c_samples, 
                    init_config.at, nsteps=run_info["nsteps"], temp=run_info[config_type]["temp"], 
                    dt=run_info[config_type]["dt"], τ=run_info[config_type]["τ"], maxp=run_info[config_type]["maxp"])
            
            plot_HMD(E_tot, E_pot, E_kin, T, P, m, k=1)

            if calc_settings["calculator"] == "DFTB"
                at, py_at = HMD.CALC.DFTB_calculator(cfgs[end], config_type, calc_settings)
            elseif calc_settings["calculator"] == "CASTEP"
                at, py_at = HMD.CALC.CASTEP_calculator(cfgs[end], config_type, calc_settings)
            elseif calc_settings["calculator"] == "NRLTB"
                at, py_at = HMD.CALC.NRLTB_calculator(cfgs[end], config_type, m)
            elseif calc_settings["calculator"] == "NRLTBpy3"
                at, py_at = HMD.CALC.NRLTBpy3_calculator(cfgs[end], config_type, calc_settings, m)
            end

            push!(al, at)
            #write_xyz("./HMD_it$(m).xyz", [py_at])

            #at = HMD.CALC.CASTEP_calculator(cfgs[end], config_type, dft_settings)
            #push!(al, at)

            #write_xyz("./HMD_it$(m).xyz", py_at)

            #write_xyz("./HMD_surf_vac/crash_$(m).xyz", cfgs[end])
            #run(`/Users/Cas/anaconda2/bin/python /Users/Cas/.julia/dev/MDLearn/HMD_surf_vac/convert.py $(m) $(config_type)`)
        end
    end
    return al
end

function run(IP, B, Vref, c_samples, at; nsteps=100, temp=100, dt=1.0, τ=0.5, maxp=0.15)
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

    cfgs = []

    running = true

    i = 2
    while running && i < nsteps
        at, p = HMD.COM.VelocityVerlet_com(Vref, B, c_samples, at, dt * HMD.MD.fs, τ=τ)
        P[i] = maximum(p)
        Ek = ((0.5 * sum(at.M) * norm(at.P ./ at.M)^2)/length(at.M)) / length(at.M)
        Ep = (energy(IP, at) - E0) / length(at.M)
        E_tot[i] = Ek + Ep
        E_pot[i] = Ep
        E_kin[i] = Ek
        T[i] = Ek / (1.5 * HMD.MD.kB)
        i+=1

        @show maximum(p)
        if maximum(p) > maxp #abs((E_tot[i-1]/E_tot[2] - 1.0)) > 0.05
            running = false
        end
        if i % 10 == 0
            τ *= 1.1
        end
        push!(cfgs, at)
    end
    
    return E_tot[1:i], E_pot[1:i], E_kin[1:i], T[1:i], P[1:i], varEs[1:i], varFs[1:i], cfgs
end

function plot_HMD(E_tot, E_pot, E_kin, T, P, i; k=50) # varEs,
    p1 = plot()
    plot!(p1,E_tot[2:end-k], label="")
    plot!(p1,E_kin[2:end-k], label="")
    plot!(p1,E_pot[2:end-k], label="")
    ylabel!(p1, "Energy (eV)")
    p2 = plot()
    plot!(p2, T[2:end-k],label="")
    ylabel!(p2, "T (K)")
    p4 = plot()
    plot!(p4, P[2:end-k],label="")
    xlabel!(p4,"MDstep")
    ylabel!(p4, "P")
    p5 = plot(p1, p2, p4, size=(400,550), layout=grid(3, 1, heights=[0.6, 0.2, 0.2]))
    savefig("./HMD_$(i).pdf")
end


end

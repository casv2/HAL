using Plots
using MDLearn
using IPFitting
using JuLIP
using ACE
using LinearAlgebra
using ASE
using JuLIP.MLIPs: SumIP

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
    savefig("./HMD_surf_vac/HMD_$(i).pdf")
end

function run_HMD(IP, B, c_samples, at; nsteps=100, temp=100, dt=1.0, τ=0.5)
    E_tot = zeros(nsteps)
    E_pot = zeros(nsteps)
    E_kin = zeros(nsteps)
    T = zeros(nsteps)
    P = zeros(nsteps)
    varEs = zeros(nsteps)
    varFs = zeros(nsteps)

    E0 = energy(IP, at)

    at = MDLearn.MD.MaxwellBoltzmann_scale(at, temp)
    at = MDLearn.MD.Stationary(at)

    cfgs = []

    Vref = OneBody(:Ti => -5.817622899211898)

    running = true

    i = 2
    while running && i < nsteps
        at, p = MDLearn.HMD.VelocityVerlet_uncertain_com(Vref, B, c_samples, at, dt * MDLearn.MD.fs, τ=τ)
        P[i] = p
        Ek = ((0.5 * sum(at.M) * norm(at.P ./ at.M)^2)/length(at.M)) / length(at.M)
        Ep = (energy(IP, at) - E0) / length(at.M)
        E_tot[i] = Ek + Ep
        E_pot[i] = Ep
        E_kin[i] = Ek
        T[i] = Ek / (1.5 * MDLearn.MD.kB)
        i+=1
        if i % 10 == 0
            @show p, abs((E_tot[i-1]/E_tot[2] - 1.0))
            if abs((E_tot[i-1]/E_tot[2] - 1.0)) > 0.02
                running = false
            end
            τ *= 1.05
            push!(cfgs, ASEAtoms(at))
        end
    end
    
    return E_tot[1:i], E_pot[1:i], E_kin[1:i], T[1:i], P[1:i], varEs[1:i], varFs[1:i], cfgs
end

function do_fit(al, R)
    r0 = rnn(:Ti)
    
    Bpair = pair_basis(species = :Ti,
        r0 = r0,
        maxdeg = 3,
        rcut = 7.0,
        pcut = 1,
        pin = 0)
    
    Bsite = rpi_basis(species = :Ti,
          N = 3,                       # correlation order = body-order - 1
          maxdeg = 10,            # polynomial degree
          r0 = r0,                      # estimate for NN distance
          #D = SparsePSHDegree(; wL=1.3, csp=1.0),
          rin = R, rcut = 5.5,   # domain for radial basis (cf documentation) #5.5
          pin = 2);

    weights = Dict(
        "ignore"=> [],
        "default" => Dict("E" => 15.0, "F" => 1.0 , "V" => 1.0 ),
        )

    Vref = OneBody(:Ti => -1586.0195)

    B = JuLIP.MLIPs.IPSuperBasis([Bpair, Bsite]);
    
    dB = LsqDB("", B, al);

    Ψ, Y = IPFitting.Lsq.get_lsq_system(dB, verbose=true,
                                Vref=Vref, Ibasis = :,Itrain = :,
                                weights=weights, regularisers = [])
    
    c_samples = MDLearn.Uncertain.do_brr(Ψ, Y, 100.0, 1.0, 5);
    
    IP = JuLIP.MLIPs.SumIP(Vref, JuLIP.MLIPs.combine(B, c_samples[:,1]))

    add_fits_serial!(IP, al, fitkey="IP")
    rmse_, rmserel_ = rmse(al; fitkey="IP");
    rmse_table(rmse_, rmserel_)
    
    return IP, B, c_samples
end

al = IPFitting.Data.read_xyz("/Users/Cas/temp/DB_3.xyz", energy_key="energy", force_key="force", virial_key="virial");
R = minimum(IPFitting.Aux.rdf(al, 4.0))
IP, B, c_samples = do_fit(al, R)

init_ats = IPFitting.Data.read_xyz(@__DIR__() * "/HMD_init_hcp_bcc_vac_surf.xyz")
iters = 5

for j in 1:length(init_ats)
    init_at = init_ats[j].at
    config_type = configtype(init_ats[j])
    if config_type in ["bcc","hcp"]
        temp = 6050 #6050
        dt = 1.0 #0.5
        τ = 0.005
    elseif config_type in ["bcc_surf", "bcc_vac"]
        temp = 3050 #2050
        dt = 1.0 #1.0
        τ = 0.001
    elseif config_type in ["hcp_surf", "hcp_vac"]
        temp = 2050 #2050
        dt = 1.0 #1.0
        τ = 0.001
    end
    for l in 1:iters
        init_config = deepcopy(init_at)
        m = (j-1)*iters + l
        @show m
        al = IPFitting.Data.read_xyz(@__DIR__() * "/DB_$(m-1).xyz", energy_key="energy", force_key="force", virial_key="virial");
        R = minimum(IPFitting.Aux.rdf(al, 4.0))
        IP, B, c_samples = do_fit(al, R)

        E_tot, E_pot, E_kin, T, P, varEs, varFs, cfgs = run_HMD(IP, B, c_samples, init_config, nsteps=10000, temp=temp, dt=dt, τ=τ);
        plot_HMD(E_tot, E_pot, E_kin, T, P, m, k=1)
        write_xyz("./HMD_surf_vac/crash_$(m).xyz", cfgs[end])
        run(`/Users/Cas/anaconda2/bin/python /Users/Cas/.julia/dev/MDLearn/HMD_surf_vac/convert.py $(m) $(config_type)`)
    end
end
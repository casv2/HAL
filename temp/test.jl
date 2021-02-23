using JuLIP
using HMD
using IPFitting
using ACE
using LinearAlgebra

al = IPFitting.Data.read_xyz("/Users/Cas/.julia/dev/HMD/DB_0.xyz", energy_key="energy", force_key="force")

Vref = OneBody(:Ti => -5.817622899211898)

start_configs = IPFitting.Data.read_xyz(@__DIR__() * "/HMD_init_hcp_bcc_vac_surf.xyz", energy_key="energy", force_key="forces")

run_info = Dict(
    "HMD_iters" => 5,
    "nsteps" => 10000,
    "ncoms" => 20,
    "bcc" => Dict("temp" => 6000, "τ" => 0.1, "dt" => 1.0),
    "hcp" => Dict("temp" => 6000, "τ" => 0.1, "dt" => 1.0),
    "bcc_surf" => Dict("temp" => 3000, "τ" => 0.05, "dt" => 1.0),
    "bcc_vac" => Dict("temp" => 3000, "τ" => 0.05, "dt" => 1.0),
    "hcp_surf" => Dict("temp" => 2000, "τ" => 0.05, "dt" => 1.0),
    "hcp_vac" => Dict("temp" => 2000, "τ" => 0.05, "dt" => 1.0),
    #"hcp" => Dict("temp" => 3000, "τ" => 0.05, "dt" => 1.0)
)

weights = Dict(
        "ignore"=> [],
        "default" => Dict("E" => 15.0, "F" => 1.0 , "V" => 1.0 ),
        )

dftb_settings = Dict(
    "kpoint_spacing" => 0.03,
    "ASE_DFTB_COMMAND" => "/Users/Cas/miniconda3/bin/dftb+ > PREFIX.out",
    "DFTB_PREFIX" => "/Users/Cas/.julia/dev/HMD/temp"
)

Binfo = Dict(
    "Z" => :Ti,
    "N" => 4,
    "deg" => 14,
    "2B" => 3,
    "r0" => rnn(:Ti),
    "Nrcut" => 5.5,
    "2Brcut" => 7.0,
)

al_HMD = HMD.RUN.run_HMD(Binfo, Vref, weights, al, start_configs, run_info, dftb_settings)

al_HMD[end].D["F"]

scatter(vcat(forces(IP, al_HMD[end].at)...), al_HMD[end].D["F"])


al_HMD[end-1].D["F"]

################################################

r0 = rnn(:Ti)

R = minimum(IPFitting.Aux.rdf(al_HMD, 4.0))

Bpair = pair_basis(species = :Ti,
    r0 = r0,
    maxdeg = 3,
    rcut = 7.0,
    pcut = 1,
    pin = 0)

Bsite = rpi_basis(species = :Ti,
        N = 3,                       # correlation order = body-order - 1
        maxdeg = 18,            # polynomial degree
        r0 = r0,                      # estimate for NN distanc
        #D = SparsePSHDegree(; wL=1.3, csp=1.0),
        rin = R, rcut = 5.5,   # domain for radial basis (cf documentation) #5.5
        pin = 2);

B = JuLIP.MLIPs.IPSuperBasis([Bpair, Bsite]);

Vref = OneBody(:Ti => -5.817622899211898)

dB = LsqDB("", B, al_HMD);

weights = Dict(
    "ignore"=> [],
    "default" => Dict("E" => 30.0, "F" => 1.0 , "V" => 1.0 ),
    "FLD_bcc" => Dict("E" => 30.0, "F" => 1.0 , "V" => 0.0 ),
    "FLD_hcp" => Dict("E" => 30.0, "F" => 1.0 , "V" => 0.0 ),
    "PH_bcc" => Dict("E" => 30.0, "F" => 30.0 , "V" => 0.0 ),
    "PH_hcp" => Dict("E" => 30.0, "F" => 30.0 , "V" => 0.0 ),
    )

Ψ, Y = IPFitting.Lsq.get_lsq_system(dB, verbose=true,
                            Vref=Vref, Ibasis = :,Itrain = :,
                            weights=weights, regularisers = [])

α, β = HMD.BRR.maxim_hyper(Ψ, Y, 1e-5, 1e-5, 10, 1e-3)

c_samples = HMD.BRR.do_brr(Ψ, Y, α, β, 20)

@show norm(c_samples[:,2])

add_fits_serial!(IP, al_HMD, fitkey="IP")
rmse_, rmserel_ = rmse(al_HMD; fitkey="IP");
rmse_table(rmse_, rmserel_)

al_test = IPFitting.Data.read_xyz("/Users/Cas/.julia/dev/MDLearn/exampleTB_com/bcc_2500_sel.xyz", energy_key="energy", force_key="force")

IP = JuLIP.MLIPs.SumIP(Vref, JuLIP.MLIPs.combine(B, c_samples[:,4]))


save_dict(@__DIR__() * "/fit.json", Dict("IP" => write_dict(IP), "info" => lsqinfo))




IP, lsqinfo = lsqfit(dB, solver=(:rid, 1.20), weights = weights, Vref = Vref, asmerrs=true);
@show norm(lsqinfo["c"])
rmse_table(lsqinfo["errors"]) 

add_fits_serial!(IP, al_test, fitkey="IP")
rmse_, rmserel_ = rmse(al_test; fitkey="IP");
rmse_table(rmse_, rmserel_)


using Plots

Es = [ at.D["E"][1]/length(at.at) for at in al ]

histogram(Es)
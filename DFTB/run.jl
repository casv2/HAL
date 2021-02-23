using JuLIP
using HMD
using IPFitting
using ACE
using LinearAlgebra

al = IPFitting.Data.read_xyz(@__DIR__() * "/Ti_LIQ_DFTB.xyz", energy_key="energy", force_key="force")

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

Vref = OneBody(:Ti => -17.682448443582274)

calc_settings = Dict(
    "calculator" => "DFTB",
    "kpoint_spacing" => 0.03,
    "ASE_DFTB_COMMAND" => "/Users/Cas/miniconda3/bin/dftb+ > PREFIX.out",
    "DFTB_PREFIX" => "/Users/Cas/.julia/dev/HMD/DFTB"
)

Binfo = Dict(
    "Z" => :Ti,
    "N" => 3,
    "deg" => 14,
    "2B" => 3,
    "r0" => rnn(:Ti),
    "Nrcut" => 5.5,
    "2Brcut" => 7.0,
)

al_HMD = HMD.RUN.run_HMD(Binfo, Vref, weights, al, start_configs, run_info, calc_settings)


#################################




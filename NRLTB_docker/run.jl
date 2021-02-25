using JuLIP
using HMD
using IPFitting
using ACE
using LinearAlgebra

al = IPFitting.Data.read_xyz(@__DIR__() * "/DB_0.xyz", energy_key="energy", force_key="force")

start_configs = IPFitting.Data.read_xyz(@__DIR__() * "/HMD_init_hcp_bcc_vac_surf.xyz", energy_key="energy", force_key="forces")[1:2]

run_info = Dict(
    "HMD_iters" => 1,
    "nsteps" => 5000,
    "ncoms" => 20,
    "bcc" => Dict("temp" => 6000, "τ" => 0.1, "dt" => 1.0, "maxp" => 0.15),
    "hcp" => Dict("temp" => 6000, "τ" => 0.1, "dt" => 1.0, "maxp" => 0.15),
    "bcc_surf" => Dict("temp" => 3000, "τ" => 0.05, "dt" => 1.0, "maxp" => 0.15),
    "bcc_vac" => Dict("temp" => 3000, "τ" => 0.05, "dt" => 1.0, "maxp" => 0.15),
    "hcp_surf" => Dict("temp" => 2000, "τ" => 0.05, "dt" => 1.0, "maxp" => 0.15),
    "hcp_vac" => Dict("temp" => 2000, "τ" => 0.05, "dt" => 1.0, "maxp" => 0.15),
    #"hcp_ad_atom" => Dict("temp" => 1000, "τ" => 5.0, "dt" => 1.0)
)

weights = Dict(
        "ignore"=> [],
        "default" => Dict("E" => 15.0, "F" => 1.0 , "V" => 1.0 ),
        )

Vref = OneBody(:Ti => -5.817622899211898)

calc_settings = Dict(
    "calculator" => "NRLTBpy3",
    "docker_id" => "2c95e0b518da",
    "docker_folder" => "/root/Work/HMD/docker",
    "real_folder" => "/Users/Cas/Work/HMD/docker",
)

Binfo = Dict(
    "Z" => :Ti,
    "N" => 3,
    "deg" => 12,
    "2B" => 3,
    "r0" => rnn(:Ti),
    "Nrcut" => 5.5,
    "2Brcut" => 7.0,
)

al_HMD = HMD.RUN.run_HMD(Binfo, Vref, weights, al, start_configs, run_info, calc_settings)

#################################




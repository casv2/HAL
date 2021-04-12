using IPFitting
using HMD
using JuLIP
using ACE
using PyCall
using LinearAlgebra
using Random
using Plots
using JuLIP.MLIPs: SumIP
using Statistics
using Distributions
using ASE

al_in = IPFitting.Data.read_xyz("/Users/Cas/Work/ACE/Si/Si.xyz", energy_key="dft_energy", force_key="dft_force")
dia_configs = filter(at -> configtype(at) == "dia", al_in)

function save_configs(al)
    al_save = []
    for at in al
        py_at = ASEAtoms(at.at)

        D_info = PyDict(py_at.po[:info])
        D_arrays = PyDict(py_at.po[:arrays])

        D_info["config_type"] = "HAL_" * configtype(at)
        D_info["energy"] = at.D["E"]
        D_info["virial"] = at.D["V"]
        D_arrays["forces"] = at.D["F"]

        py_at.po[:info] = D_info
        py_at.po[:arrays] = D_arrays

        push!(al_save, py_at)
    end
    write_xyz("HAL_it.xyz", al)
end

configtype.(dia_configs)

save_configs(dia_configs)

using PyCall

PyVector()

al_save = []
for at in dia_configs[1:3]
    py_at = ASEAtoms(at.at)

    D_info = PyDict(py_at.po[:info])
    D_arrays = PyDict(py_at.po[:arrays])

    D_info["config_type"] = "HAL_" * configtype(at)
    D_info["energy"] = at.D["E"]
    D_info["virial"] = at.D["V"]
    D_arrays["forces"] = reshape(at.D["F"], length(at.at), 3)

    py_at.po[:info] = D_info
    py_at.po[:arrays] = D_arrays

    push!(al_save, py_at.po)
end

write_xyz("temp.xyz", al_save)

py_write("./temp.xyz", PyVector(al_save))

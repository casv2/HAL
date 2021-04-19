module CALC

using PyCall
using ASE
using IPFitting: Dat
using IPFitting
using LinearAlgebra
EAM = pyimport("ase.calculators.eam")["EAM"]
CASTEP = pyimport("ase.calculators.castep")["Castep"]
DFTB = pyimport("ase.calculators.dftb")["Dftb"]
ORCA = pyimport("ase.calculators.orca")["ORCA"]
try
    #VASP = pyimport("ase.calculators.vasp")["Vasp"]
    VASP = pyimport("vasp_gr")["VASP"]
catch
    VASP = pyimport("ase.calculators.vasp")["Vasp"]
end
try DFTB = pyimport("quippy.potential")["Potential"] catch end

function EAM_calculator(at, config_type)
    py_at = ASEAtoms(at)

    calculator = EAM(potential=@__DIR__() * "/Ti1.eam.fs")
    py_at.po[:set_calculator](calculator)

    E = py_at.po.get_potential_energy()
    F = py_at.po.get_forces()
    #V = -1.0 * py_at.get_stress() * py_at.get_volume()

    D_info = PyDict(py_at.po[:info])
    D_arrays = PyDict(py_at.po[:arrays])

    D_info["config_type"] = "HMD_" * config_type
    D_info["energy"] = E
    D_arrays["force"] = F

    py_at.po[:info] = D_info
    py_at.po[:arrays] = D_arrays

    dat = Dat( at,"HMD_" * config_type, E = E, F = F)#, V = V)

    return dat, py_at
end

function NRLTB_calculator(at, config_type, m)
    py_at = ASEAtoms(at)

    write_xyz("/Users/Cas/.julia/dev/HMD/NRLTB/crash_$(m).xyz", py_at)
    run(`/Users/Cas/anaconda2/bin/python /Users/Cas/.julia/dev/HMD/NRLTB/convert.py $(m)`)
    #V = -1.0 * py_at.get_stress() * py_at.get_volume()

    al = IPFitting.Data.read_xyz("/Users/Cas/.julia/dev/HMD/NRLTB/crash_conv_$(m).xyz", energy_key="energy", force_key="forces")
    E = al[1].D["E"]
    F = al[1].D["F"]

    dat = Dat( at,"HMD_" * config_type, E = E, F = F)#, V = V)

    return dat, py_at
end

function NRLTBpy3_calculator(at, config_type, calc_settings, m)
    py_at = ASEAtoms(at)

    docker_folder = calc_settings["docker_folder"] 
    docker_id = calc_settings["docker_id"] 
    real_folder = calc_settings["real_folder"] 

    write_xyz(calc_settings["real_folder"]  * "/crash_$(m).xyz", py_at)

    run(`$(real_folder)/bash_conv.sh $(docker_id) $(docker_folder)/convert.py $(m) $(docker_folder)`)

    al = IPFitting.Data.read_xyz("$(real_folder)/crash_conv_$(m).xyz", energy_key="energy", force_key="forces")
    E = al[1].D["E"]
    F = al[1].D["F"]
    #V = -1.0 * py_at.get_stress() * py_at.get_volume()

    dat = Dat( at,"HMD_" * config_type, E = E, F = F)#, V = V)

    return dat, py_at
end

function CASTEP_calculator(at, config_type, calc_settings)
    py_at = ASEAtoms(at)

    calculator = CASTEP()
    calculator[:_castep_command] = calc_settings["_castep_command"]
    calculator[:_directory] =calc_settings["_directory"]
    calculator[:_castep_pp_path] = calc_settings["_castep_pp_path"]
    calculator.param[:cut_off_energy] = calc_settings["cut_off_energy"]
    calculator.param[:calculate_stress] = calc_settings["calculate_stress"]
    calculator.param[:smearing_width] = calc_settings["smearing_width"]
    calculator.param[:finite_basis_corr] = calc_settings["finite_basis_corr"]
    calculator.param[:mixing_scheme] = calc_settings["mixing_scheme"]
    calculator.param[:write_checkpoint] = calc_settings["write_checkpoint"]
    #calculator.cell[:kpoints_mp_spacing] = 0.1
    calculator.cell[:kpoint_mp_spacing] = calc_settings["kpoint_mp_spacing"]
    calculator.param[:fine_grid_scale] = calc_settings["fine_grid_scale"]
    py_at.po[:set_calculator](calculator)

    E = py_at.po.get_potential_energy(force_consistent=true)
    F = py_at.po.get_forces()
    V = -1.0 * py_at.po.get_stress(voigt=false) * py_at.po.get_volume()

    dat = Dat( at, "HMD_" * config_type, E = E, F = F, V = V)

    D_info = PyDict(py_at.po[:info])
    D_arrays = PyDict(py_at.po[:arrays])

    D_info["config_type"] = "HMD_" * config_type
    D_info["energy"] = E
    D_info["virial"] = V
    D_arrays["force"] = F

    py_at.po[:info] = D_info
    py_at.po[:arrays] = D_arrays

    return dat, py_at
end

function ORCA_calculator(at, config_type, calc_settings)
    py_at = ASEAtoms(at)

    calculator = ORCA(label=calc_settings["label"],
    orca_command=calc_settings["orca_command"],
    charge=calc_settings["charge"], mult=calc_settings["mult"], task=calc_settings["task"],
    orcasimpleinput=calc_settings["orcasimpleinput"],
    orcablocks=calc_settings["orcablocks"])
    py_at.po[:set_calculator](calculator)

    E = py_at.po.get_potential_energy()
    F = py_at.po.get_forces()

    dat = Dat( at, "HMD_" * config_type, E = E, F = F)

    D_info = PyDict(py_at.po[:info])
    D_arrays = PyDict(py_at.po[:arrays])

    D_info["config_type"] = "HMD_" * config_type
    D_info["energy"] = E
    D_arrays["forces"] = F

    py_at.po[:info] = D_info
    py_at.po[:arrays] = D_arrays

    return dat, py_at
end

function DFTB_calculator(at, config_type, calc_settings)
    py_at = ASEAtoms(at)

    PyCall.PyDict(PyCall.pyimport("os").environ)["ASE_DFTB_COMMAND"] = calc_settings["ASE_DFTB_COMMAND"]
    PyCall.PyDict(PyCall.pyimport("os").environ)["DFTB_PREFIX"] = calc_settings["DFTB_PREFIX"]

    kpoint_spacing = calc_settings["kpoint_spacing"]

    kspace = norm.(eachrow(at.cell)) .^ -1
    kpoints = vcat(floor.(Int, kspace ./ kpoint_spacing)...)

    @show kpoints

    calculator = DFTB(at=py_at.po,
                    label="Ti",
                    kpts=kpoints,
                    Hamiltonian_="DFTB",
                    Hamiltonian_SCC="Yes",
                    Hamiltonian_SCCTolerance=1e-8,
                    Hamiltonian_Filling_="Fermi",
                    Hamiltonian_Filling_Temperature=300)
    
    py_at.po[:set_calculator](calculator)

    E = py_at.po.get_potential_energy()
    F = py_at.po.get_forces()
    #V = -1.0 * py_at.po.get_stress(voigt=false) * py_at.po.get_volume()

    dat = Dat( at, "HMD_" * config_type, E = E, F = F)#, V = V)

    D_info = PyDict(py_at.po[:info])
    D_arrays = PyDict(py_at.po[:arrays])

    D_info["config_type"] = "HMD_" * config_type
    D_info["energy"] = E
    D_arrays["force"] = F
    #D_info["virial"] = V

    py_at.po[:info] = D_info
    py_at.po[:arrays] = D_arrays

    return dat, py_at
end

function VASP_calculator(at, config_type, calc_settings)
    VASP = pyimport("vasp_gr")["VASP"]

    py_at = ASEAtoms(at)

    PyCall.PyDict(PyCall.pyimport("os").environ)["VASP_PP_PATH"] = calc_settings["VASP_PP_PATH"]

    calculator = VASP(
        command=calc_settings["command"],
        xc=calc_settings["xc"],
        directory=calc_settings["directory"],
        setups=calc_settings["setups"],
        prec=calc_settings["prec"],
    )

    py_at.po[:set_calculator](calculator)

    E = py_at.po.get_potential_energy()
    F = py_at.po.get_forces()
    V = -1.0 * py_at.po.get_stress(voigt=false) * py_at.po.get_volume()

    dat = Dat( at, "HAL_" * config_type, E = E, F = F, V = V)

    D_info = PyDict(py_at.po[:info])
    D_arrays = PyDict(py_at.po[:arrays])

    D_info["config_type"] = "HAL_" * config_type
    D_info["energy"] = E
    D_info["virial"] = V
    D_arrays["force"] = F

    py_at.po[:info] = D_info
    py_at.po[:arrays] = D_arrays

    return dat, py_at
end

end

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
Aims = pyimport("ase.calculators.aims")["Aims"]
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

    D_info["config_type"] = "HAL_" * config_type
    D_info["energy"] = E
    D_arrays["force"] = F

    py_at.po[:info] = D_info
    py_at.po[:arrays] = D_arrays

    dat = Dat( at,"HAL_" * config_type, E = E, F = F)#, V = V)

    return dat, py_at
end

function NRLTB_calculator(at, config_type, m)
    py_at = ASEAtoms(at)

    write_xyz("/Users/Cas/.julia/dev/HAL/NRLTB/crash_$(m).xyz", py_at)
    run(`/Users/Cas/anaconda2/bin/python /Users/Cas/.julia/dev/HAL/NRLTB/convert.py $(m)`)
    #V = -1.0 * py_at.get_stress() * py_at.get_volume()

    al = IPFitting.Data.read_xyz("/Users/Cas/.julia/dev/HAL/NRLTB/crash_conv_$(m).xyz", energy_key="energy", force_key="forces")
    E = al[1].D["E"]
    F = al[1].D["F"]

    dat = Dat( at,"HAL_" * config_type, E = E, F = F)#, V = V)

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

    dat = Dat( at,"HAL_" * config_type, E = E, F = F)#, V = V)

    return dat, py_at
end

function Aims_calculator(at, config_type, calc_settings)
    py_at = ASEAtoms(at)

    calculator = Aims()
    for (key, value) in calc_settings
        if key ∉ ["calculator"]
            pycall(calculator."parameters"."__setattr__", Nothing, key, value)
        end
    end

    py_at.po[:set_calculator](calculator)

    E = py_at.po.get_potential_energy(force_consistent=true)
    F = py_at.po.get_forces()

    if at.pbc[1] == true
        V = -1.0 * py_at.po.get_stress(voigt=false) * py_at.po.get_volume()
        dat = Dat( at, "HAL_" * config_type, E = E, F = vcat(F'...), V = V')
    else
        dat = Dat( at, "HAL_" * config_type, E = E, F = vcat(F'...) )
    end    

    D_info = PyDict(py_at.po[:info])
    D_arrays = PyDict(py_at.po[:arrays])

    D_info["config_type"] = "HAL_" * config_type
    D_info["energy"] = E    
    if at.pbc[1] == true
        D_info["virial"] = V
    end
    D_arrays["forces"] = F

    py_at.po[:info] = D_info
    py_at.po[:arrays] = D_arrays

    return dat, py_at
end

function CASTEP_calculator(at, config_type, calc_settings)
    py_at = ASEAtoms(at)

    calculator = CASTEP()
    calculator[:_castep_command] = calc_settings["_castep_command"]
    #calculator[:_castep_pp_path] = calc_settings["_castep_pp_path"]
    calculator[:_directory] =calc_settings["_directory"]
    calculator.cell[:kpoint_mp_spacing] = calc_settings["kpoint_mp_spacing"]

    #calculator[:_castep_pp_path] = calc_settings["_castep_pp_path"]
    for (key, value) in calc_settings
        if key ∉ ["calculator", "_castep_command", "_directory", "kpoint_mp_spacing"]
            calculator.param[Symbol(key)] = value
        end
    end

    py_at.po[:set_calculator](calculator)

    E = py_at.po.get_potential_energy(force_consistent=true)
    F = py_at.po.get_forces()
    V = -1.0 * py_at.po.get_stress(voigt=false) * py_at.po.get_volume()

    dat = Dat( at, "HAL_" * config_type, E = E, F = vcat(F'...), V = V')

    D_info = PyDict(py_at.po[:info])
    D_arrays = PyDict(py_at.po[:arrays])

    D_info["config_type"] = "HAL_" * config_type
    D_info["energy"] = E
    D_info["virial"] = V
    D_arrays["forces"] = F

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

    dat = Dat( at, "HAL_" * config_type, E = E, F = vcat(F'...))

    D_info = PyDict(py_at.po[:info])
    D_arrays = PyDict(py_at.po[:arrays])

    D_info["config_type"] = "HAL_" * config_type
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

    dat = Dat( at, "HAL_" * config_type, E = E, F = F)#, V = V)

    D_info = PyDict(py_at.po[:info])
    D_arrays = PyDict(py_at.po[:arrays])

    D_info["config_type"] = "HAL_" * config_type
    D_info["energy"] = E
    D_arrays["force"] = F
    #D_info["virial"] = V

    py_at.po[:info] = D_info
    py_at.po[:arrays] = D_arrays

    return dat, py_at
end

function VASP_calculator(at, config_type, i, j, calc_settings)
    #VASP = pyimport("vasp_gr")["Vasp"]
    VASP = pyimport("ase.calculators.vasp")["Vasp"]

    py_at = ASEAtoms(at)

    PyCall.PyDict(PyCall.pyimport("os").environ)["VASP_PP_PATH"] = calc_settings["VASP_PP_PATH"]

    VASP_dir = joinpath(calc_settings["directory"], "HAL_$(i)_$(j)")

    if isdir(VASP_dir)
        rm(VASP_dir,recursive=true)
    end

    mkdir(VASP_dir)
    cp("KPGEN", joinpath(VASP_dir, "KPGEN"), force=true)

    if calc_settings["setups"] != "recommended"
        setups = PyDict()
        for (key,value) in calc_settings["setups"]
            setups[key] = value
        end
    else
        setups = "recommended"
    end

    calculator = VASP(
        command=calc_settings["command"],
        xc=calc_settings["xc"],
        directory=VASP_dir,
        setups=setups,
        prec=calc_settings["prec"],
        encut=calc_settings["encut"],
        ismear=calc_settings["ismear"],
        sigma=calc_settings["sigma"],
        #ENCUT=calc_settings["ENCUT"],
        #ENCUT=calc_settings["ENCUT"],
    )

    calculator[:write_input](py_at.po)

    return nothing
    # py_at.po[:set_calculator](calculator)

    # E = py_at.po.get_potential_energy()
    # F = py_at.po.get_forces()
    # V = -1.0 * py_at.po.get_stress(voigt=false) * py_at.po.get_volume()

    # dat = Dat( at, "HAL_" * config_type, E = E, F = F, V = V)

    # D_info = PyDict(py_at.po[:info])
    # D_arrays = PyDict(py_at.po[:arrays])

    # D_info["config_type"] = "HAL_" * config_type
    # D_info["energy"] = E
    # D_info["virial"] = V
    # D_arrays["force"] = F

    # py_at.po[:info] = D_info
    # py_at.po[:arrays] = D_arrays

    # return dat, py_at
end

end

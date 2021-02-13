module CALC

using PyCall
using ASE
using IPFitting: Dat
EAM = pyimport("ase.calculators.eam")["EAM"]
CASTEP = pyimport("ase.calculators.castep")["Castep"]

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

    dat = Dat( at,"HMD", E = E, F = F)#, V = V)

    return dat, py_at
end

function CASTEP_calculator(at, config_type, dft_settings)
    py_at = ASEAtoms(at)

    calculator = CASTEP()
    calculator[:_castep_command] = dft_settings["_castep_command"]
    calculator[:_directory] = dft_settings["_directory"]
    calculator.param[:cut_off_energy] = dft_settings["cut_off_energy"]
    calculator.param[:calculate_stress] = dft_settings["calculate_stress"]
    calculator.param[:smearing_width] = dft_settings["smearing_width"]
    calculator.param[:finite_basis_corr] = dft_settings["finite_basis_corr"]
    calculator.param[:mixing_scheme] = dft_settings["mixing_scheme"]
    calculator.param[:write_checkpoint] = dft_settings["write_checkpoint"]
    #calculator.cell[:kpoints_mp_spacing] = 0.1
    calculator.cell[:kpoint_mp_spacing] = dft_settings["kpoint_mp_spacing"]
    calculator.cell[:fine_grid_scale] = dft_settings["fine_grid_scale"]
    py_at.po[:set_calculator](calculator)

    E = py_at.po.get_potential_energy()
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

end

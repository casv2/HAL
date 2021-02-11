module CALC

using PyCall
using ASE
using IPFitting: Dat
EAM = pyimport("ase.calculators.eam")["EAM"]
CASTEP = pyimport("ase.calculators.castep")["Castep"]

function EAM_calculator(at)
    py_at = ASEAtoms(at).po

    calculator = EAM(potential=@__DIR__() * "/Ti1.eam.fs")
    py_at[:set_calculator](calculator)

    E = py_at.get_potential_energy()
    F = py_at.get_forces()
    #V = -1.0 * py_at.get_stress() * py_at.get_volume()

    dat = Dat( at,"HMD", E = E, F = F)#, V = V)

    return dat
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
    calculator.cell[:kpoint_mp_grid] = dft_settings["kpoint_mp_grid"]
    py_at.po[:set_calculator](calculator)

    E = py_at.po.get_potential_energy()
    F = py_at.po.get_forces()
    V = -1.0 * py_at.po.get_stress(voigt=false) * py_at.po.get_volume()

    write_xyz("./temp.xyz", py_at)

    dat = Dat( at, "HMD_" * config_type, E = E, F = F, V = V)

    return dat
end

end

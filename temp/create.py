from ase.calculators.eam import EAM
from ase.lattice.cubic import BodyCenteredCubic
from ase.io import read, write

calculator = EAM(potential="../src/Ti1.eam.fs")

al = []

for i in range(100):
    at = BodyCenteredCubic("Ti", latticeconstant=3.16) * (2,2,2)
    at.rattle(0.1)
    at.set_calculator(calculator)
    at.get_potential_energy()
    at.get_forces()
    at.info["config_type"] = "EAM_rattle"
    al.append(at)

write("EAM_start.xyz", al)
from quippy.potential import Potential
from ase.io import read, write
import sys
import os

i = sys.argv[1]
docker_folder = sys.argv[2]

calculator = Potential("TB NRL-TB", param_filename=os.path.join(docker_folder, "quip_params.xml"))

at = read(os.path.join(docker_folder, "crash_{}.xyz".format(i)))
at.set_calculator(calculator)

at.arrays["force"] = at.get_forces()
at.info["energy"] = at.get_potential_energy()
at.info["config_type"] = "HMD_iter"

write(os.path.join(docker_folder, "crash_conv_{}.xyz".format(i)), at)



# i = sys.argv[1]

# calculator = quippy.Potential("TB NRL-TB", param_filename="/Users/Cas/.julia/dev/MDLearn/exampleTB/quip_params.xml")

# at = read("/Users/Cas/.julia/dev/MDLearn/exampleTB/crash_{}.xyz".format(i))
# at.set_calculator(calculator)

# at.arrays["force"] = at.get_forces()
# at.info["energy"] = at.get_potential_energy()
# at.info["config_type"] = "HMD_iter{}".format(i)

# write("/Users/Cas/.julia/dev/MDLearn/exampleTB/crash_conv_{}.xyz".format(i), at)

#i0 = int(i) - 1

#al = read("/Users/Cas/.julia/dev/MDLearn/exampleTB/DB_{}.xyz".format(i0), ":")
#al.append(at)
#write("/Users/Cas/.julia/dev/MDLearn/exampleTB/DB_{}.xyz".format(i), al)

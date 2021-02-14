using HMD
using JuLIP
using IPFitting
using ACE

al = IPFitting.Data.read_xyz(@__DIR__() * "/EAM_start.xyz", energy_key="energy", force_key="forces")

r0 = rnn(:Ti)
    
Bpair = pair_basis(species = :Ti,
    r0 = r0,
    maxdeg = 3,
    rcut = 7.0,
    pcut = 1,
    pin = 0)

Bsite = rpi_basis(species = :Ti,
        N = 3,                       # correlation order = body-order - 1
        maxdeg = 10,            # polynomial degree
        r0 = r0,                      # estimate for NN distance
        #D = SparsePSHDegree(; wL=1.3, csp=1.0),
        rin = 0.7*r0, rcut = 5.5,   # domain for radial basis (cf documentation) #5.5
        pin = 2);

weights = Dict(
    "ignore"=> [],
    "default" => Dict("E" => 15.0, "F" => 1.0 , "V" => 1.0 ),
    )

Vref = OneBody(:Ti => -5.0)

B = JuLIP.MLIPs.IPSuperBasis([Bpair, Bsite]);

start_configs = IPFitting.Data.read_xyz(@__DIR__() * "/HMD_init.xyz", energy_key="energy", force_key="forces")

run_info = Dict(
    "HMD_iters" => 5,
    "nsteps" => 10000,
    "ncoms" => 20,
    "bcc" => Dict("temp" => 3000, "τ" => 1e20, "dt" => 0.00005),
    "hcp" => Dict("temp" => 3000, "τ" => 1e20, "dt" => 0.00005)
)

weights = Dict(
        "ignore"=> [],
        "default" => Dict("E" => 15.0, "F" => 1.0 , "V" => 1.0 ),
        )

dft_settings = Dict(
    "_castep_command" => "mpirun -n 8 /home/vc381/castep/castep.mpi",
    "_directory" => "./_CASTEP",
    "cut_off_energy" => 400,
    "calculate_stress" => true,
    "smearing_width" => 0.1,
    "finite_basis_corr" => "automatic",
    "mixing_scheme" => "Pulay",
    "write_checkpoint" => "none",
    "kpoint_mp_grid" => "1 1 1"
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

Bsite = rpi_basis(species = Binfo["Z"],
                N = Binfo["N"],       # correlation order = body-order - 1
                maxdeg = Binfo["deg"],  # polynomial degree
                r0 = Binfo["r0"],     # estimate for NN distance
                rin = R, rcut = Binfo["Nrcut"],   # domain for radial basis (cf documentation)
                pin = 2) 

Bpair = pair_basis(species = Binfo["Z"],
    r0 = Binfo["r0"],
    maxdeg = Binfo["2B"],
    rcut = Binfo["2Brcut"],
    pcut = 1,
    pin = 0)   # pin = 0 means no inner cutoff

B = JuLIP.MLIPs.IPSuperBasis([Bpair, Bsite]);

HMD.RUN.run_HMD(B, Vref, weights, al, start_configs, run_info, dft_settings)




using ASE

ASEAtoms(at)

py_at = ASEAtoms(at)
using IPFitting: Dat
EAMp = pyimport("ase.calculators.eam")["EAM"]

calculator = EAMp(potential="/Users/Cas/.julia/dev/HMD/src/Ti1.eam.fs")
py_at.po[:set_calculator](calculator)

py_at.po.rattle(0.5)
E = py_at.po.get_potential_energy()
F = py_at.po.get_forces()
S = py_at.po.get_stress()

py_at.po.get_volume()

write_xyz("./temp.xyz", py_at)
#V = -1.0 * py_at.get_stress() * py_at.get_volume()
@show E
@show F

at

dat = Dat(at, "HMD", E = E, F = F)

Dat(at)

1:run_info["HMD_iters"]
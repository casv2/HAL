using ACE
using IPFitting
using ColorSchemes
using PyCall
using JuLIP
using PrettyTables

function get_lml(N, deg, Binfo, Vref, weights, al)
    BRR = pyimport("sklearn.linear_model")["BayesianRidge"]

    clf = BRR(compute_score=true)

    R = minimum(IPFitting.Aux.rdf(al, 4.0))

    Bsite = rpi_basis(species = Binfo["Z"],
        N = N,       # correlation order = body-order - 1
        maxdeg = deg,  # polynomial degree
        r0 = Binfo["r0"],     # estimate for NN distance
        rin = R, rcut = Binfo["Nrcut"],   # domain for radial basis (cf documentation)
        pin = 2)

    Bpair = pair_basis(species = Binfo["Z"],
        r0 = Binfo["r0"],
        maxdeg = Binfo["2B"],
        rcut = Binfo["2Brcut"],
        pcut = 1,
        pin = 0)

    B = JuLIP.MLIPs.IPSuperBasis([Bpair, Bsite]);

    dB = LsqDB("", B, al);

    Ψ, Y = IPFitting.Lsq.get_lsq_system(dB, verbose=true,
                                Vref=Vref, Ibasis = :,Itrain = :,
                                weights=weights, regularisers = [])

    clf.fit(Ψ, Y)

    score = clf.scores_
    c = clf.coef_

    @show c

    @show score

    return score[end]
end

function find_N_deg_table(Binfo, Vref, weights, al; Ns=[2,3,4,5], degs=[8,9,10,11])
    _lml = zeros(length(degs), length(Ns))
    _D = Dict()

    for (i,N) in enumerate(Ns)
        for (j,deg) = enumerate(degs)
            _lml[j,i] = round(get_lml(N, deg, Binfo, Vref, weights, al), digits=2)
            _D[(N, deg)] = _lml[j,i]
        end
    end

    _, (maxN, maxdeg) = findmax(_D)

    hl = Highlighter((lml,i,j)->true,
                            (h,lml,i,j)->begin
                                color = get(colorschemes[:coolwarm], lml[i,j], (minimum(_lml),maximum(_lml)))
                                return Crayon(foreground = (round(Int,color.r*255),
                                                            round(Int,color.g*255),
                                                            round(Int,color.b*255)))
                            end)

    pretty_table(_lml, Ns, row_names = degs, highlighters = hl)

    return _lml
end

E0 = -158.54496821
r0 = rnn(:Si)

al_in = IPFitting.Data.read_xyz("./Si/Si.xyz", auto=true)
dia_configs = filter(at -> configtype(at) == "dia", al_in)
#liq_configs = filter(at -> configtype(at) == "liq", al_in)

#al = vcat(dia_configs, liq_configs)

Vref = OneBody(:Si => E0)

Binfo = Dict(
    "Z" => :Si,
    "N" => 3,
    "deg" => 12,
    "2B" => 3,
    "r0" => rnn(:Si),
    "Nrcut" => 5.5,
    "2Brcut" => 7.0,
)

weights = Dict(
        "default" => Dict("E" => 15.0, "F" => 1.0 , "V" => 1.0 ),
        )

_lml = find_N_deg_table(Binfo, Vref, weights, dia_configs; Ns=[2], degs=[6,8,10]) #,14,16,18]
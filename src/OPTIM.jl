module OPTIM

using ACE
using IPFitting
using ColorSchemes
using PrettyTables
using HMD

function get_lml(N, deg, Binfo, Vref, weights, al)

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

    α, β, c, lml_score = HMD.BRR.maxim_hyper(Ψ, Y)

    return lml_score
end

function find_N_deg_table(Binfo, Vref, weights, al; Ns=[2,3,4,5], degs=[3,4,5,6,7])
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

function find_N_deg(Binfo, Vref, weights, al; Ns=[2], degs=[3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20])
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

    M = []
    lml0 = -1e10
    for deg in degs
        lml = round(get_lml(maxN, deg, Binfo, Vref, weights, al), digits=2)
        @show lml
        if lml > lml0
            lml0 = lml
            maxdeg = deg
        else
            break
        end
    end

    return maxN, maxdeg
end

end
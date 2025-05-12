import Adapt

using CUDA

struct GF
    rid::Int32
    eid::Int32
    g::Matrix{Float32}
end

Adapt.@adapt_structure GF

struct Rec
    rid::Int32
    obs::Vector{Float32}
end

Adapt.@adapt_structure Rec

nr = 2
ne = 3
npts = 10
dt = 0.1

rs_cpu = Rec[]
for i = 1:nr
    push!(rs_cpu, Rec(i, randn(npts)))
end

rs_gpu = cu(rs_cpu)

gs_cpu = GF[]
GIDCS = CartesianIndices((ne, nr))
for igc = eachindex(GIDCS[:])
    I = GIDCS[igc]
    push!(gs_cpu, GF(I[2], I[1], randn(npts, 6)))
end
gs_gpu = cu(gs_cpu)

ep = randn(6, ne)
ep_gpu = cu(ep)

function kernel!(result, gs, rs, ep)
    CUDA.@sync begin
        igf = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        gf = gs[igf]
        rec = rs[gf.rid]
        x = 0.0
        a = 0.0
        b = 0.0
        for i = eachindex(rec.obs)
            a += rec.obs[i] * rec.obs[i]
            w = 0.0
            for j = 1:6
                w += gf.g[i, j] * ep[j, gf.eid]
            end
            b += w * w
            x += rec.obs[i] * w
        end
        result[igf] = x / sqrt(a*b)
    end
    return nothing
end

mis = zeros(Float32, length(gs_cpu))
mis_gpu = cu(mis)

@cuda threads=length(gs_cpu) kernel!(mis_gpu, gs_gpu, rs_gpu, ep_gpu)

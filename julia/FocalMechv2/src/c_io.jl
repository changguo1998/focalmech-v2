struct GlobalSetting
    tag::String

    n_record::Int64
    n_event_location::Int64
    # preprocessing
    low_frequency::Vector{Float64}
    high_frequency::Vector{Float64}
    data_prefix::Float64
    data_suffix::Float64

    # SDR
    dstrike::Float64
    ddip::Float64
    drake::Float64
end

const MAX_INVERSION_TAG_LENGTH = 128

struct GlobalSetting_C
    tag::String
    n_record::Int64
    n_event_location::Int64
    n_frequency_pairs::Int64
    dstrike::Float64
    ddip::Float64
    drake::Float64
end

function GlobalSetting_C(s::GlobalSetting)
    return GlobalSetting_C(
        s.tag, s.n_record, s.n_event_location,
        length(s.low_frequency) * length(s.high_frequency),
        s.dstrike, s.ddip, s.drake
    )
end

function read_global_setting_from_database(io::IO)
    cbuf = zeros(UInt8, MAX_INVERSION_TAG_LENGTH)
    read!(io, cbuf)
    n_record = read(io, Int64)
    n_event_location = read(io, Int64)
    n_frequency_pairs = read(io, Int64)
    dstrike = read(io, Float64)
    ddip = read(io, Float64)
    drake = read(io, Float64)
    sbuf = String(filter(>(0), cbuf))
    return GlobalSetting_C(sbuf, n_record, n_event_location, n_frequency_pairs, dstrike, ddip, drake)
end

function write_to_database(io::IO, s::GlobalSetting_C)
    cbuf = zeros(UInt8, MAX_INVERSION_TAG_LENGTH)
    cbuf[1:length(s.tag)] .= UInt8.(collect(s.tag))
    write(io, cbuf)
    write(io, s.n_record)
    write(io, s.n_event_location)
    write(io, s.n_frequency_pairs)
    write(io, s.dstrike)
    write(io, s.ddip)
    write(io, s.drake)
    return nothing
end

struct Record_C
    id::Int64
    data::Matrix{Float64}
    phase::Vector{Tuple{Int,Int,Bool}}
end

function Record_C(
    id::Integer,
    data::AbstractMatrix{<:Real},
    phase::AbstractVector{Tuple{Int,Int,Bool}}=Tuple{Int,Int,Bool}[])
    return Records(Int64(id), Float64.(data), phase)
end

function read_record_from_database(io::IO, gs::GlobalSetting_C)
    id = read(io, Int64)
    npts = read(io, Int64)
    nphase = read(io, Int64)
    data = zeros(Float64, npts, gs.n_frequency_pairs)
    read!(io, data)
    phase = Vector{Tuple{Int,Int,Bool}}(undef, nphase)
    for i = 1:nphase
        i1 = read(io, Int64)
        i2 = read(io, Int64)
        i3 = read(io, UInt8)
        phase[i] = (i1, i2, Bool(i3))
    end
    return Record_C(id, data, phase)
end

function write_to_database(io::IO, r::Record_C)
    write(io, r.id)
    write(io, Int64(size(r.data, 1)))
    write(io, Int64(length(r.phase)))
    write(io, r.data)
    for p in r.phase
        write(io, Int64(p[1]))
        write(io, Int64(p[2]))
        write(io, UInt8(p[3]))
    end
    return nothing
end

function pp_records_in_different_frequency(
    data::Vector{Float64},
    freqL::Vector{Float64},
    freqH::Vector{Float64},
    dt::Float64)
    nL = length(freqL)
    nH = length(freqH)
    n_pair = nL * nH
    data_filtered = zeros(length(data), n_pair)
    for iL = eachindex(freqL), iH = eachindex(freqH)
        data_filtered[:, iH+(iL-1)*nH] = SeisTools.DataProcess.bandpass(data, freqL[iL], freqH[iH], 1.0 / dt)
    end
    return data_filtered
end

struct GreenFun_C
    record_id::Int64
    event_location_id::Int64
    g11::Matrix{Float64}
    g22::Matrix{Float64}
    g33::Matrix{Float64}
    g12::Matrix{Float64}
    g13::Matrix{Float64}
    g23::Matrix{Float64}
end

function read_green_fun_from_database(io::IO, rs::Vector{Record_C}, gs::GlobalSetting_C)
    rid = read(io, Int64)
    eid = read(io, Int64)
    idx = findfirst(r->r.id == rid, rs)
    npts = size(rs[idx].data, 1)
    g11 = zeros(Float64, npts, gs.n_frequency_pairs)
    read!(io, g11)
    g22 = zeros(Float64, npts, gs.n_frequency_pairs)
    read!(io, g22)
    g33 = zeros(Float64, npts, gs.n_frequency_pairs)
    read!(io, g33)
    g12 = zeros(Float64, npts, gs.n_frequency_pairs)
    read!(io, g12)
    g13 = zeros(Float64, npts, gs.n_frequency_pairs)
    read!(io, g13)
    g23 = zeros(Float64, npts, gs.n_frequency_pairs)
    read!(io, g23)
    return GreenFun_C(rid, eid, g11, g22, g33, g12, g13, g23)
end

function write_to_database(io::IO, gf::GreenFun_C)
    write(io, gf.record_id)
    write(io, gf.event_location_id)
    write(io, gf.g11)
    write(io, gf.g22)
    write(io, gf.g33)
    write(io, gf.g12)
    write(io, gf.g13)
    write(io, gf.g23)
    return nothing
end

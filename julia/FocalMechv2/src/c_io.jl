struct GlobalSetting
    tag::String

    n_record::Int64
    n_event_location::Int64
    n_phase::Int64
    # preprocessing
    low_frequency::Vector{Float64}
    high_frequency::Vector{Float64}
    data_prefix::Float64
    data_suffix::Float64

    # SDR
    nstrike::Int64
    ndip::Int64
    nrake::Int64
    dstrike::Float64
    ddip::Float64
    drake::Float64
end

const MAX_INVERSION_TAG_LENGTH = 128

struct GlobalSetting_C
    tag::String
    n_record::Int64
    n_event_location::Int64
    n_frequency_pair::Int64
    n_phase::Int64
    nstrike::Int64
    ndip::Int64
    nrake::Int64
    dstrike::Float64
    ddip::Float64
    drake::Float64
end

function GlobalSetting_C(s::GlobalSetting)
    return GlobalSetting_C(
        s.tag, s.n_record, s.n_event_location,
        length(s.low_frequency) * length(s.high_frequency),
        s.n_phase, s.nstrike, s.ndip, s.nrake,
        s.dstrike, s.ddip, s.drake
    )
end

function GlobalSetting_C(io::IO)
    cbuf = zeros(UInt8, MAX_INVERSION_TAG_LENGTH)
    read!(io, cbuf)
    n_record = read(io, Int64)
    n_event_location = read(io, Int64)
    n_frequency_pair = read(io, Int64)
    n_phase = read(io, Int64)
    nstrike = read(io, Int64)
    ndip = read(io, Int64)
    nrake = read(io, Int64)
    dstrike = read(io, Float64)
    ddip = read(io, Float64)
    drake = read(io, Float64)
    sbuf = String(filter(>(0), cbuf))
    return GlobalSetting_C(sbuf, n_record, n_event_location, n_frequency_pair, n_phase,
        nstrike, ndip, nrake, dstrike, ddip, drake)
end

function write_to_database(io::IO, s::GlobalSetting_C)
    cbuf = zeros(UInt8, MAX_INVERSION_TAG_LENGTH)
    cbuf[1:length(s.tag)] .= UInt8.(collect(s.tag))
    write(io, cbuf)
    write(io, s.n_record)
    write(io, s.n_event_location)
    write(io, s.n_frequency_pair)
    write(io, s.n_phase)
    write(io, s.nstrike)
    write(io, s.ndip)
    write(io, s.nrake)
    write(io, s.dstrike)
    write(io, s.ddip)
    write(io, s.drake)
    return nothing
end

struct Record_C
    id::Int64
    data::Matrix{Float64}
end

function Record_C(
    id::Integer,
    data::AbstractMatrix{<:Real})
    return Records(Int64(id), Float64.(data))
end

function Record_C(io::IO, gs::GlobalSetting_C)
    id = read(io, Int64)
    npts = read(io, Int64)
    data = zeros(Float64, npts, gs.n_frequency_pair)
    read!(io, data)
    return Record_C(id, data)
end

function write_to_database(io::IO, r::Record_C)
    write(io, r.id)
    write(io, Int64(size(r.data, 1)))
    write(io, r.data)
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

struct GreenFunction_C
    record_id::Int64
    event_location_id::Int64
    g11::Matrix{Float64}
    g22::Matrix{Float64}
    g33::Matrix{Float64}
    g12::Matrix{Float64}
    g13::Matrix{Float64}
    g23::Matrix{Float64}
end

function GreenFunction_C(io::IO, rs::Vector{Record_C}, gs::GlobalSetting_C)
    rid = read(io, Int64)
    eid = read(io, Int64)
    idx = findfirst(r -> r.id == rid, rs)
    npts = size(rs[idx].data, 1)
    g11 = zeros(Float64, npts, gs.n_frequency_pair)
    read!(io, g11)
    g22 = zeros(Float64, npts, gs.n_frequency_pair)
    read!(io, g22)
    g33 = zeros(Float64, npts, gs.n_frequency_pair)
    read!(io, g33)
    g12 = zeros(Float64, npts, gs.n_frequency_pair)
    read!(io, g12)
    g13 = zeros(Float64, npts, gs.n_frequency_pair)
    read!(io, g13)
    g23 = zeros(Float64, npts, gs.n_frequency_pair)
    read!(io, g23)
    return GreenFunction_C(rid, eid, g11, g22, g33, g12, g13, g23)
end

function write_to_database(io::IO, gf::GreenFunction_C)
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

struct Phase_C
    rid::Int64
    eid::Int64
    type::Int64
    Rstart::Int64
    Estart::Int64
    length::Int64
    flag::Bool
end

function Phase_C(io::IO)
    _rid = read(io, Int64)
    _eid = read(io, Int64)
    _type = read(io, Int64)
    _Rstart = read(io, Int64)
    _Estart = read(io, Int64)
    _L = read(io, Int64)
    _flag = read(io, UInt8)
    return Phase_C(_rid, _eid, _type, _Rstart, _Estart, _L, Bool(_flag))
end

function write_to_database(io::IO, ps::Phase_C)
    write(io, ps.rid)
    write(io, ps.eid)
    write(io, ps.type)
    write(io, ps.Rstart)
    write(io, ps.Estart)
    write(io, ps.length)
    write(io, UInt8(ps.flag))
    return nothing
end

struct Result_C
    n_freq::Int64
    n_phase::Int64
    n_fm::Int64
    waveform::Array{Float64, 3}
    shift::Array{Int64, 3}
    polarity::Array{Float64, 3}
    ps_ratio::Array{Float64, 3}
end

function Result_C(io::IO)
    nfreq = read(io, Int64)
    np = read(io, Int64)
    nfm = read(io, Int64)
    waveform = zeros(Float64, nfreq, np, nfm)
    read!(io, waveform)
    shift = zeros(Int64, nfreq, np, nfm)
    read!(io, shift)
    polarity = zeros(Float64, nfreq, np, nfm)
    read!(io, polarity)
    ps_ratio = zeros(Float64, nfreq, np, nfm)
    read!(io, ps_ratio)
    return Result_C(nfreq, np, nfm, waveform, shift, polarity, ps_ratio)
end

function write_to_database(io::IO, r::Result_C)
    write(io, r.n_freq)
    write(io, r.n_phase)
    write(io, r.n_fm)
    write(io, r.waveform)
    write(io, r.shift)
    write(io, polarity)
    write(io, ps_ratio)
    return nothing
end

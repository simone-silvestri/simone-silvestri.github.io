getconfig(f::FFTPlan) = f.forward_plan, f.backward_plan

function ∂x(u, fftplan)
    fft, ifft = getconfig(fftplan)
    storage = fftplan.storage[1]
    k = fftplan.grid.kx

    storage .= u
    fft * storage
    storage .*= im .* k
    ifft * storage
    return real(storage)
end

function ∂y(v, fftplan)
    fft, ifft = getconfig(fftplan)
    storage = fftplan.storage[1] 
    k = fftplan.grid.ky

    storage .= v
    fft * storage
    storage .*= im .* k
    ifft * storage
    return real(storage)
end

function ∂z(w, fftplan)
    fft, ifft = getconfig(fftplan)
    storage = fftplan.storage[1]
    k = fftplan.grid.kz

    storage .= w
    fft * storage
    storage .*= im .* k
    ifft * storage
    return real(storage)
end

function Δ(u, fftplan)
    fft, ifft = getconfig(fftplan)
    storage = fftplan.storage[1]
    k = fftplan.grid.kv

    storage .= u
    fft * storage
    storage .*= im .* k
    ifft * storage
    return - real(storage)
end

function compute_pressure!(state, fftplan, Δt) # In fourier domain!!
    u, v, w, p = state
    fft  = fftplan.forward_plan
    grid = fftplan.grid
    su, sv, sw     = fftplan.storage
    kx, ky, kz, kv = (grid.kx, grid.ky, grid.kz, grid.kv)

    su .= u
    fft * su
    sv .= v
    fft * sv
    sw .= w
    fft * sw

    @. p = - (su * kx + sv * ky + sw * kz) / kv / Δt

    CUDA.@allowscalar p[1, 1, 1] = 0

    return nothing
end

function correct_velocities!(state, fftplan, Δt)
    (; u, v, w, p) = state    
    fft, ifft = getconfig(fftplan)
    storage = fftplan.storage[1]
    grid = fftplan.grid
    kx, ky, kz = (grid.kx, grid.ky, grid.kz)

    storage .= p .* kx
    ifft * storage
    u .= u + real(storage) .* Δt

    storage .= p .* ky
    ifft * storage
    v .= v + real(storage) .* Δt

    storage .= p .* kz
    ifft * storage
    w .= w + real(storage) .* Δt

    return nothing
end

using FFTW
using KernelAbstractions
using KernelAbstractions: CPU, @kernel, @index
using CUDA
import Base: zeros, Complex, size

const α = (0,    -17//60, -5//12)
const β = (8//15,  5//12,  3//4)

struct GPU end

GPU() = CUDA.CUDABackend(always_inline = true)

struct Grid{A, FT, K}
    Nx :: Int
    Ny :: Int
    Nz :: Int
    Lx :: FT
    Ly :: FT
    Lz :: FT
    kx :: K
    ky :: K
    kz :: K
    kv :: K

    Grid{A}(Nx::Int, Ny::Int, Nz::Int, Lx::FT, Ly::FT, Lz::FT, kx::K, ky::K, kz::K, kv::K) where {A, FT, K} = 
        new{A, FT, K}(Nx, Ny, Nz, Lx, Ly, Lz, kx, ky, kz, kv)
end

Complex(grid) = Complex{eltype(grid)} 

function Grid(arch = CPU(), FT::DataType = Float64; size, length)
    Nx, Ny, Nz = size
    Lx, Ly, Lz = convert.(Ref(FT), length)

    kx = zeros(FT, arch, (Nx, Ny, Nz))
    ky = zeros(FT, arch, (Nx, Ny, Nz))
    kz = zeros(FT, arch, (Nx, Ny, Nz))
    kv = zeros(FT, arch, (Nx, Ny, Nz))

    k1 = fftfreq(Nx, Nx / Lx) * 2π
    k2 = fftfreq(Ny, Ny / Ly) * 2π
    k3 = fftfreq(Nz, Nz / Lz) * 2π

    _compute_frequencies!(arch, (16, 16), (Nx, Ny, Nz))(kv, kx, ky, kz, k1, k2, k3)

    return Grid{typeof(arch)}(Nx, Ny, Nz, 
                              Lx, Ly, Lz,
                              kx, ky, kz, kv)
end

@kernel function _compute_frequencies!(kv, kx, ky, kz, k1, k2, k3)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        kx[i, j, k] = k1[i]
        ky[i, j, k] = k2[j]
        kz[i, j, k] = k3[k]
        kv[i, j, k] = k1[i]^2 + k2[j]^2 + k3[k]^2
    end
end

eltype(::Grid{A, FT}) where {A, FT} = FT
size(grid::Grid) = (grid.Nx, grid.Ny, grid.Nz)

zeros(grid::Grid) = zeros(eltype(grid), grid)
zeros(FT, grid::Grid{<:CPU}) = zeros(FT, CPU(), size(grid))
zeros(FT, grid::Grid{<:GPU}) = zeros(FT, GPU(), size(grid))
zeros(FT, ::CPU, dims) = zeros(FT, dims)
zeros(FT, ::GPU, dims) = CuArray(zeros(FT, dims))

struct NavierStokes{T, G, S, R, F, V}
    grid :: G
    state :: S
    rhs :: R
    fftplan :: F
    timestepper :: T
    viscosity :: V
end   

struct RungeKutta3{F, S}
    first_pressure :: F
    second_pressure :: S
end

RungeKutta3() = RungeKutta3(nothing, nothing)

struct FFTPlan{F, B, S, G}
    forward_plan :: F
    backward_plan :: B
    storage :: S
    grid :: G
end

function FFTPlan(grid)
    storage = Tuple(zeros(Complex(grid), grid) for i in 1:3)
    forward_plan  =  plan_forward_transform(storage[1], [1, 2, 3])
    backward_plan = plan_backward_transform(storage[1], [1, 2, 3])

    return FFTPlan(forward_plan, backward_plan, storage, grid)
end

plan_forward_transform(A::Array, dims, planner_flag=FFTW.PATIENT) = 
    length(dims) == 0 ? nothing : FFTW.plan_fft!(A, dims, flags=planner_flag)

plan_backward_transform(A::Array, dims, planner_flag=FFTW.PATIENT) = 
    length(dims) == 0 ? nothing : FFTW.plan_ifft!(A, dims, flags=planner_flag)

plan_forward_transform(A::CuArray, dims, planner_flag) = 
    length(dims) == 0 ? nothing : CUDA.CUFFT.plan_fft!(A, dims)

plan_backward_transform(A::CuArray, dims, planner_flag) = 
    length(dims) == 0 ? nothing : CUDA.CUFFT.plan_ifft!(A, dims)

function NavierStokes(; grid, timestepper = RungeKutta3(), Re = 10000)
    u  = zeros(grid)
    v  = zeros(grid)
    w  = zeros(grid)
    p  = zeros(Complex(grid), grid)
    rhs1 = (; u = zeros(grid), v = zeros(grid), w = zeros(grid))
    rhs2 = (; u = zeros(grid), v = zeros(grid), w = zeros(grid))
    fftplan = FFTPlan(grid)
    return NavierStokes(grid, (; u, v, w, p), (rhs1, rhs2), fftplan, timestepper, 1 / Re)
end

function time_step!(ns::NavierStokes, Δt)
    ts = ns.timestepper 

    # Substep 1
    compute_rhs!(ns, ns.rhs[1])
    advance_velocities!(ns, Δt, β[1], α[1])
    compute_pressure!(ns, ts.first_pressure, Δt)
    correct_velocities!(ns, Δt)

    # Substep 2
    compute_rhs!(ns, ns.rhs[2])
    advance_velocities!(ns, Δt, α[2], β[2])
    compute_pressure!(ns, ts.second_pressure, Δt)
    correct_velocities!(ns, Δt)

    # Substep 3
    compute_rhs!(ns, ns.rhs[1])
    advance_velocities!(ns, Δt, β[3], α[3])
    compute_pressure!(ns, nothing, Δt)
    correct_velocities!(ns, Δt)
    return nothing
end

function compute_rhs!(ns, rhs)

    compute_advection!(rhs, ns.state, ns.fftplan)

    ru, rv, rw = (rhs.u, rhs.v, rhs.w)
    u, v, w    = (ns.state.u, ns.state.v, ns.state.w)

    compute_diffusion!(ru, u, ns.viscosity, ns.fftplan)
    compute_diffusion!(rv, v, ns.viscosity, ns.fftplan)
    compute_diffusion!(rw, w, ns.viscosity, ns.fftplan)
    return nothing
end

include("derivatives.jl")

function compute_advection!(rhs, state, fftplan)

    u, v, w    = (state.u, state.v, state.w)
    ru, rv, rw = (rhs.u, rhs.v, rhs.w)
    compute_conservative_advection!(ru, u, (; u, v, w), fftplan)
    compute_conservative_advection!(rv, v, (; u, v, w), fftplan)
    compute_conservative_advection!(rw, w, (; u, v, w), fftplan)

    fft, ifft = getconfig(fftplan)
    s1, s2, s3 = fftplan.storage
    (; kx, ky, kz) = fftplan.grid
    
    s1 .= u
    fft * s1
    s2 .= s1 .* im .* kx # du/dx
    ifft * s2

    ru .-=        u .* real(s2)
    rv .-= 0.5 .* v .* real(s2)
    rw .-= 0.5 .* w .* real(s2)
    
    s2 .= s1 .* im .* ky # du/dy
    ifft * s2

    ru .-= 0.5 .* v .* real(s2)
    
    s2 .= s1 .* im .* kz # du/dz
    ifft * s2

    ru .-= 0.5 .* w .* real(s2)
    
    s1 .= v
    fft * s1
    s2 .= s1 .* im .* ky # dv/dy
    ifft * s2

    ru .-= 0.5 .* u .* real(s2)
    rv .-=        v .* real(s2)
    rw .-= 0.5 .* w .* real(s2)

    s2 .= s1 .* im .* kx # dv/dx
    ifft * s2

    rv .-= 0.5 .* u .* real(s2)
    
    s2 .= s1 .* im .* kz # dv/dz
    ifft * s2

    rv .-= 0.5 .* w .* real(s2)

    s1 .= w
    fft * s1
    s2 .= s1 .* im .* kz # dw/dz
    ifft * s2

    ru .-= 0.5 .* u .* real(s2)
    rv .-= 0.5 .* v .* real(s2)
    rw .-=        w .* real(s2)


    s2 .= s1 .* im .* kx # dw/dx
    ifft * s2

    rw .-= 0.5 .* u .* real(s2)
    
    s2 .= s1 .* im .* ky # dw/dy
    ifft * s2

    rw .-= 0.5 .* v .* real(s2)

end


function compute_conservative_advection!(rhs, vel, U, fftplan)

   rhs  .= - ∂x(vel .* U.u, fftplan)
   rhs .+= - ∂y(vel .* U.v, fftplan)
   rhs .+= - ∂z(vel .* U.w, fftplan)

    return nothing
end

compute_diffusion!(rhs, u, ν, fftplan) = rhs .+= ν .* Δ(u, fftplan)

function advance_velocities!(ns, Δt, α, β)
    @. ns.state.u = ns.state.u + Δt * (α * ns.rhs[1].u + β .* ns.rhs[2].u)
    @. ns.state.v = ns.state.v + Δt * (α * ns.rhs[1].v + β .* ns.rhs[2].v)
    @. ns.state.w = ns.state.w + Δt * (α * ns.rhs[1].w + β .* ns.rhs[2].w)
end

compute_pressure!(ns, ::Nothing,  Δt) = compute_pressure!(ns.state, ns.fftplan, Δt)
correct_velocities!(ns, Δt) = correct_velocities!(ns.state, ns.fftplan, Δt)

@kernel function _taylor_vortex_init!(state, grid)
    i, j, k = @index(Global, NTuple)

    Δx = grid.Lx / grid.Nx
    Δy = grid.Ly / grid.Ny
    Δz = grid.Lz / grid.Nz
    
    fx = 2π * Δx * i / grid.Lx
    fy = 2π * Δy * j / grid.Ly
    fz = 2π * Δz * k / grid.Lz

    state.u[i, j, k] =  sin(fx) * cos(fy) * cos(fz)
    state.v[i, j, k] = -cos(fx) * sin(fy) * cos(fz)
    state.w[i, j, k] =  0.0
end
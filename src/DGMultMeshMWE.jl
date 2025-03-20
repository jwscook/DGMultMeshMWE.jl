using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi
using Triangulate # https://juliageometry.github.io/Triangulate.jl/stable/examples/
using StartUpDG # see the examples for the origin of this script https://github.com/jlchan/StartUpDG.jl/blob/213a75c40b7ef3e6d4185fb79a20b009ea324509/src/physical_frame_basis.jl#L151
using Plots

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

"""
    initial_condition_mach_flow(x, t, equations::CompressibleEulerEquations2D)

Compressible Euler setup for a Mach N wind tunnel flow with a forward facing step.
Results in strong shock interactions as well as Kelvin-Helmholtz instabilities at later times.
See Section IV b on the paper below for details.

- Paul Woodward and Phillip Colella (1984)
  The Numerical Simulation of Two-Dimensional Fluid Flows with Strong Shocks.
  [DOI: 10.1016/0021-9991(84)90142-6](https://doi.org/10.1016/0021-9991(84)90142-6)
"""
@inline function initial_condition_mach_flow(x, t, equations::CompressibleEulerEquations2D)
    # set the freestream flow parameters
    rho_freestream = 1.4
    v1 = 5.0
    v2 = 0.0
    p_freestream = 1.0

    prim = SVector(rho_freestream, v1, v2, p_freestream)
    return prim2cons(prim, equations)
end

initial_condition = initial_condition_mach_flow
boundary_condition_inflow = BoundaryConditionDirichlet(initial_condition_mach_flow)

# Outflow boundary condition.
# FIXME: For now only works for supersonic outflow where all values come from the internal state.
# The bones are here for the subsonic outflow as well. One simply needs to pass the reference pressure
# to set it from the outside, the rest comes from the internal solution.
# Once fixed this function should probably move to `compressible_euler_2d.jl`
# See the reference below for a discussion on inflow/outflow boundary conditions.
#
# - Jan-Reneé Carlson (2011)
#   Inflow/Outflow Boundary Conditions with Application to FUN3D.
#   [NASA TM 20110022658](https://ntrs.nasa.gov/citations/20110022658)
@inline function boundary_condition_outflow(u_inner, normal_direction::AbstractVector, x, t,
                                            surface_flux_function,
                                            equations::CompressibleEulerEquations2D)
    # # This would be for the general case where we need to check the magnitude of the local Mach number
    # norm_ = norm(normal_direction)
    # # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
    # normal = normal_direction / norm_

    # # Rotate the internal solution state
    # u_local = Trixi.rotate_to_x(u_inner, normal, equations)

    # # Compute the primitive variables
    # rho_local, v_normal, v_tangent, p_local = cons2prim(u_local, equations)

    # # Compute local Mach number
    # a_local = sqrt( equations.gamma * p_local / rho_local )
    # Mach_local = abs( v_normal / a_local )
    # if Mach_local <= 1.0
    #   p_local = # Set to the external reference pressure value (somehow? maybe stored in `equations`)
    # end

    # # Create the `u_surface` solution state where the local pressure is possibly set from an external value
    # prim = SVector(rho_local, v_normal, v_tangent, p_local)
    # u_boundary = prim2cons(prim, equations)
    # u_surface = Trixi.rotate_from_x(u_boundary, normal, equations)

    # Compute the flux using the appropriate mixture of internal / external solution states
    # flux = Trixi.flux(u_surface, normal_direction, equations)

    # NOTE: Only for the supersonic outflow is this strategy valid
    # Calculate the boundary flux entirely from the internal solution state
    flux = Trixi.flux(u_inner, normal_direction, equations)

    return flux
end



boundary_condition_convergence_test = BoundaryConditionDirichlet(initial_condition)

boundary_conditions = Dict(:left=> boundary_condition_inflow,
                           :right=> boundary_condition_outflow,
                           :lower => boundary_condition_slip_wall,
                           :upper => boundary_condition_slip_wall)

function mymesh(;polydeg=3, maxarea=0.1)
  LX = -2.0
  UX = 5.0
  LY = 0.0
  UY = 5.0
#  T0 = 0.0
#  T1 = 1.0
  
  function mytriangulation(maxarea=0.1)
    triin = Triangulate.TriangulateIO()
    triin.pointlist = Matrix{Cdouble}([LX LY; UX LY; UX UY; LX UY]')
    triin.segmentlist = Matrix{Cint}([1 2; 2 3; 3 4; 4 1]')
    triin.segmentmarkerlist = Vector{Int32}([1, 2, 3, 4])
    #triin.pointlist = Matrix{Cdouble}([LX LY; T0 LY; T0 T1; T1 T1; T1 LY; UX LY; UX UY; LX UY]')
    #triin.segmentlist = Matrix{Cint}([1 2; 2 3; 3 4; 4 5; 5 6; 6 7; 7 8; 8 1]')
    #triin.segmentmarkerlist = Vector{Int32}([1, 2, 3, 4, 5, 6, 7, 8])
    (triout, vorout) = triangulate("pa$(maxarea)DQ", triin)
    return (triout, vorout, maxarea)
  end
  
  function verticesandmap(maxarea=0.1)
    triout, varout, area = mytriangulation(maxarea)
    VX, VY = (triout.pointlist[i,:] for i = 1:size(triout.pointlist,1))
    EToV = permutedims(triout.trianglelist)
    return (VX, VY), EToV
  end
  
  rd = RefElemData(Tri(), polydeg)
  vertices, EToV = verticesandmap(maxarea)
  md = MeshData(vertices, EToV, rd)
  
  leftboundary(xy) = isapprox(xy[1], LX)
  rightboundary(xy) = isapprox(xy[1], UX)
  upperboundary(xy) = isapprox(xy[2], UY)
  function lowerboundary(xy)
    atol = 1e-8 * UY
    x, y = xy
    isapprox(y, LY, atol=atol) && return true
#    if isapprox(y, LY, atol=atol) && ((x <= T0) || (x >= T1)) # sides of hat
#      return true
#    elseif isapprox(y, T1, atol=atol) && (T0 <= x <= T1) # top of hat
#      return true
#    elseif isapprox(x, T0, atol=atol) && (T0 <= y <= T1) # left of hat
#      return true
#    elseif isapprox(x, T1, atol=atol) && (T0 <= y <= T1) # right of hat
#      return true
#    end
    return false
  end
  
  boundary_identifiers = Dict(:left => leftboundary, :lower => lowerboundary,
                              :right => rightboundary, :upper => upperboundary)
  shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max = 0.5,
                                            alpha_min = 0.001,
                                            alpha_smooth = true,
                                            variable = density_pressure)
  volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                   volume_flux_dg = volume_flux,
                                                   volume_flux_fv = surface_flux)
  solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
                 volume_integral = volume_integral)

  surface_integral = SurfaceIntegralWeakForm(FluxLaxFriedrichs())
 
  dg = Trixi.DGMulti(polydeg=polydeg, element_type=Tri(),
                     surface_flux=surface_flux,
                     surface_integral=surface_integral,
                     volume_integral=volume_integral) # Tri is from NodesAndModes.jl

  mesh = DGMultiMesh(dg, vertices, EToV; is_on_boundary=boundary_identifiers)
  
  return (mesh, dg)
end


volume_flux = flux_ranocha
surface_flux = flux_lax_friedrichs

polydeg = 4
basis = LobattoLegendreBasis(polydeg)
shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max = 0.5,
                                            alpha_min = 0.001,
                                            alpha_smooth = true,
                                            variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

mesh, dg = mymesh(;polydeg=polydeg, maxarea=0.1)
solver = dg

###############################################################################
# how to get the boundaries for the cache?

#elements = Trixi.init_elements(mesh, equations, dg.basis, Float64, Float64)
#boundaries = Trixi.init_boundaries(mesh, elements)

# need to create the cache here somehow that is the correct thing needed in
# src/solvers/dgsem_unstructured/sort_boundary_conditions.jl
# function initialize!(boundary_types...)
# the boundaries object on cache must have a iterable object, name
# and an indexable object neighbor_ids...
# Obviously, the following isn't right but it helped be get to the right error message
cache = (; boundaries=(),Trixi.create_cache(mesh, equations, solver, Float64, Float64)...)
_boundary_conditions = Trixi.digest_boundary_conditions(boundary_conditions, mesh, solver,
                                                  cache)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100

analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_restart = SaveRestartCallback(interval = 100,
                                   save_final_restart = true)

save_solution = SaveSolutionCallback(dt = 0.01,
                                     save_initial_solution = true,
                                     save_final_solution = true)

stepsize_callback = StepsizeCallback(cfl = 0.9)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_restart,
                        save_solution,
                        stepsize_callback,
                        )

###############################################################################
# run the simulation

stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (5.0e-6, 5.0e-6),
                                                     variables = (Trixi.density, pressure))
sol = solve(ode, SSPRK43(stage_limiter!);
            dt = 1e-17, # gets bigger
            ode_default_options()...,
            saveat=0:0.1:tspan[2],
            callback = callbacks);


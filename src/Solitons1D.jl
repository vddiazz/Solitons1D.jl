module Solitons1D

using JLD2
using LaTeXStrings
using LinearAlgebra
using NPZ
using Plots
using SpecialFunctions

include("ff_dynamics.jl")
export ff_kak
export ff_origin

include("ff_plot.jl")
export ff_anim

include("moduli_solve.jl")
export eq_import
export moduli_RK4_m2
export F_kak
export U_kak_phi4
export W_kak_phi4
export m2_step
export moduli_RK4_nm2

include("aux.jl")
export profile_kak_m2
export energy_m2

end

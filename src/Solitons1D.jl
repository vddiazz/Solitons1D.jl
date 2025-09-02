module Solitons1D

include("ff_dynamics.jl")
export ff_kak
export ff_origin

include("ff_plot.jl")
export ff_anim

include("moduli_solve.jl")
export eq_import
export moduli_RK4_m2

end

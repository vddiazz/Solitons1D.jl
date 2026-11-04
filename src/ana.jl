using LinearAlgebra
using Serialization
using JLD2
using SpecialFunctions
using ProgressMeter
using Base.Threads
using LoopVectorization
using Interpolations
using LsqFit
using Statistics

### analsis of models

function period_moduli(path::String,model::String,moduli::String,dt::Float64,dx::Float64,vs::Array{Float64})

	T = zeros(Float64, length(vs))

	N = Int64(150/dt)
	
	println()
	@showprogress 1 "Computing times..." for (idx,vin) in enumerate(vs)

		ts = Float64[]
		a = npzread("$(path)/a_v=$(-vin)_dt=$(dt)_dx=$(dx).npy")

		for t in 2:N-1
			if a[t-1] > a[t] && a[t+1] > a[t] && a[t] < 0
				push!(ts,t*dt)  
			end
		end

		if length(ts) >= 2
			T[idx] = ts[2]-ts[1]
		else
			T[idx] = 333e-7
		end
	end

	return T

end

function period_ff(path::String,model::String,dt::Float64,dx::Float64,vs::Array{Float64})

	T = zeros(Float64, length(vs))

	N = Int64(150/dt)

	println()
	@showprogress 1 "Computing times..." for (idx,vin) in enumerate(vs)

		ts = Float64[]
		f = npzread("$(path)/kak_ff_origin_v=$(vin).npy")

		for t in 2:N-1
			if f[t-1] > f[t] && f[t+1] > f[t] && f[t] < 0
				push!(ts,t*dt)  
			end
		end

		if length(ts) >= 2
			T[idx] = ts[2]-ts[1]
		else
			T[idx] = 333e-7
		end
	end

	return T

end

clean(x) = (x isa Real && !isfinite(x)) ? nothing : x

function omega_flight_shape(a::Array{Float64},B::Array{Float64})

	omega_0 = sqrt(3)

	dt = 0.0005
	N = length(a)	

	t_vals = collect(0:dt:150-dt)

	# extract slice
	
	ts = Int64[]

	for t in 2:N
		if a[t-1]*a[t] < 0
			push!(ts,t)
		end
	end

	ti = ts[2]
	tf = ts[3]
	
	# slice arrays

	t_slice = t_vals[ti:tf] .- ti*dt
	B_slice = B[ti:tf]

	# fit
    
	model(t_slice,coef) = coef[1] .+ coef[2].*t_slice .+ coef[3].*cos.(coef[4].*t_slice .+ coef[5])
   
	# checks
	m  = (t_vals .>= ti*dt) .& (t_vals .<= tf*dt)
	println()
	println("npts = ", count(m), "  length = ", tf*dt - ti*dt, "  periods ≈ ", (tf*dt - ti*dt)/(2*pi/1.66))
	println("extrema of B: ", extrema(B[m]), "  any NaN: ", any(!isfinite, B[m]))
	println() 

	best = nothing
    for omega_test in (0.8*omega_0, omega_0, 1.2*omega_0), phi_test in (0, pi/2, pi, 3*pi/2)
        f = curve_fit(model, t_slice, B_slice, [mean(B_slice), 0.0, (maximum(B_slice)-minimum(B_slice))/2, omega_test, phi_test])
        (best === nothing || sum(f.resid.^2) < sum(best.resid.^2)) && (best = f)
	end

	p  = best.param
    se = try stderror(best) catch; fill(NaN, 5) end

	return Dict(
        "B0"    => clean(p[1]), "B0_err"    => clean(se[1]),
        "B1"    => clean(p[2]), "B1_err"    => clean(se[2]),
        "B2"    => clean(p[3]), "B2_err"    => clean(se[3]),
        "omega" => clean(p[4]), "omega_err" => clean(se[4]),
        "phi"   => clean(p[5]), "phi_err"   => clean(se[5]),
    )
end







using LinearAlgebra
using Serialization
using JLD2
using SpecialFunctions
using ProgressMeter
using Base.Threads
using LoopVectorization
using Interpolations

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

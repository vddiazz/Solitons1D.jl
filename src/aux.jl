using LinearAlgebra
using JLD2
using Plots
using SpecialFunctions
using LoopVectorization

#-------------------- energy

function energy_m2(field_m,field,field_p,out)
    # field_m : field profile at t-1
    # field : field profile at t
    # field_p : field profile at t+1

    dt = 0.0001
    dx = 0.01

    Dx = zeros(Float64, length(field)-2)
    for j in 1:1:length(Dx)
        Dx[j] = (field[j+2]-field[j])/(2*dx)
     end

    Dt = (field_p - field_m)/(2*dt)
    deleteat!(Dt,1)
    deleteat!(Dt,length(Dt))
    
    Edens = zeros(Float64, length(Dx))

    for j in 1:1:length(Dx)
        Edens[j] = 0.5*Dt[j]^2 - 0.5*Dx[j]^2 - (1-field[j]^2)^2
    end

    E = sum(Edens)*dx

    return E
end

function energy_moduli(model::String,moduli::String,x::Array{Float64},M0,dM0)
    
    #--- params
	nm 	= length(M0)
    N 	= length(M0[1])
    dx 	= x[2]-x[1]

	gamma = 1/sqrt(1-dM0[1][1]^2)

    #--- metric and potential
	g 	= zeros(Float64, N,nm,nm)
    T 	= zeros(Float64, N,nm,nm)
	V	= zeros(Float64, N)
	E 	= zeros(Float64, N) 

	println()
	@showprogress 1 "Computing energy..." for t in 1:N
		e 		= zeros(Float64, nm,length(x))
		Vdens 	= zeros(Float64, length(x))

		Mtemp 	= [M0[i][t] for i in 1:nm] 
		dMtemp 	= [dM0[i][t] for i in 1:nm] 
		
		# densities

    	@inbounds @fastmath for idx in 1:length(x)
        	e[:,idx] 	.= ForwardDiff.gradient(M -> F_kak(model,moduli,x[idx],M,gamma), Mtemp)
			Vdens[idx] 	= W_kak(model,moduli,x[idx],Mtemp,gamma)	
		end		

		# metric, kinetic energy and potential energy

		@inbounds @fastmath for j in 1:nm, i in 1:nm
    		g[t,i,j] = sum(e[i,:] .* e[j,:])*dx
			T[t,i,j] = 0.5*g[t,i,j]*dMtemp[i]*dMtemp[j]
		end		
	
		V[t] = sum(Vdens)*dx

		E[t] = sum(T[t,:,:]) + V[t]

	end
	println()

	return g,T,V,E	
end


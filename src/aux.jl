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

function pot_kak_m2(model::String,moduli::String,space::Array{Float64},a,B,gamma)

	dx = space[2] - space[1]
    M = [a,B]

    Vdens = zeros(Float64, length(space))

    @inbounds for idx in eachindex(space)
        Vdens[idx] = W_kak(model,moduli,space[idx],M,gamma)
    end

    return sum(Vdens)*dx
end

function energy_moduli(model::String,moduli::String,space::Array{Float64},M0,dM0)
    
    #--- params
	nm 	= length(M0)
    N 	= length(M0[1])
    dx 	= space[2]-space[1]

	#vs = range(0.0, 0.5, length=50)
	#gamma_vals = 1 ./ sqrt.(1 .- vs .^ 2) 
	#gamma = 2.294157338705618 # v=0.9
	#gamma = 7.088812050083354 # v=0.99
	
	# gamma selection
	if moduli == "aB" || moduli == "pR" || moduli == "pR2"
		gamma = 888.
	elseif moduli == "maB" || moduli == "mpR" ||moduli == "mpR2" || moduli == "mmpR2"
		gamma = 1/sqrt(1-dM0[1][1]^2)
		#gamma = gamma_vals[50] # do gamma_idx+1
	end

    #--- metric and potential
	g 	= zeros(Float64, N,nm,nm)
    T 	= zeros(Float64, N,nm,nm)
	V	= zeros(Float64, N)
	E 	= zeros(Float64, N) 

	V_inf0 = pot_kak_m2(model,moduli,space,10.,0.,gamma)

	if moduli == "aB" || moduli == "maB"
	
		V_a = zeros(Float64, N)
		V_B =  zeros(Float64, N)
		V_int = zeros(Float64, N)

		println()
		@showprogress 1 "Computing energy..." for t in 1:N
			e 		= zeros(Float64, nm,length(space))
			Vdens 	= zeros(Float64, length(space))

			Mtemp 	= [M0[i][t] for i in 1:nm] 
			dMtemp 	= [dM0[i][t] for i in 1:nm] 
		
			# densities
    		@inbounds @fastmath for idx in 1:length(space)
        		e[:,idx] 	.= ForwardDiff.gradient(M -> F_kak(model,moduli,space[idx],M,gamma), Mtemp)
				Vdens[idx] 	= W_kak(model,moduli,space[idx],Mtemp,gamma)	
			end		

			# metric, kinetic energy

			@inbounds @fastmath for j in 1:nm, i in 1:nm
    			g[t,i,j] = sum(e[i,:] .* e[j,:])*dx
				T[t,i,j] = 0.5*g[t,i,j]*dMtemp[i]*dMtemp[j]
			end		
	
			V[t] = sum(Vdens)*dx

			V_a[t] = pot_kak_m2(model,moduli,space,M0[1][t],0.,gamma) - V_inf0
			V_B[t] = pot_kak_m2(model,moduli,space,10.,M0[2][t],gamma) - V_inf0
			V_int[t] = V[t] - V_a[t] - V_B[t] - V_inf0
	
			E[t] = sum(T[t,:,:]) + V[t]

		end
	
		return g,T,V,V_a,V_B,V_int,[V_inf0],E
	
	else
		# m3
	end
end


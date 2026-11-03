using LinearAlgebra
using ForwardDiff
using SpecialFunctions
using LoopVectorization

#-------------------- omega

function omega_kink(model,moduli,M0)

	n = length(M0)
	dx = 0.1


	# coefficient functions
	e = zeros(Float64, n,length(x))
	ddV = zeros(Float64, n,n,length(x))
 
	@threads for idx in 1:length(x)
		e[:,idx] .= ForwardDiff.gradient(M -> F_kak(model,moduli,x[idx],M,gamma), M0)

		ddV[:,:,idx] .= ForwardDiff.hessian(M -> W_kak(model,moduli,x[idx],M,gamma), M0)
	end

	# compute metric
	M = zeros(Float64, n,n)

	for j in 1:n
		for i in 1:n
			M[i,j] = sum()

	# compute hessian
	K = zeros(Float64, n,n)

#------------------------- field formulae

function F_kink(model,moduli,x, M, gamma)
    if model == "phi4"
	if moduli == "aB"
            f = tanh(x-M[1]) + M[2]*( sinh(x-M[1])/(cosh(x-M[1]))^2)
        elseif moduli == "maB"
	    #
	elseif moduli == "pR"
	    #
	elseif moduli == "mpR"
	    #
	elseif moduli == "pR2"
	    a = M[1]; C1 = M[2]; C2 = M[3]
	    f = tanh(x-a) + C1*(x-a)/(cosh(x-a))^2 - C2*(x-a)^2*tanh(x-a)/cosh(x-a)^2
	elseif moduli == "aBg"
	    #
	end
    end
    return f
end

function U_kink(model,moduli, x, M,gamma)
    if model == "phi4"
        U = 0.5*(1-F_kink(model,moduli,x,M,gamma)^2)^2
    end
    return U
end

function F_kak(model,moduli,x, M, gamma)
    if model == "phi4"
        if moduli == "aB"
            f = tanh(x+M[1]) - tanh(x-M[1]) - 1 + (M[2]/tanh(M[1]))*( sinh(x+M[1])/(cosh(x+M[1]))^2 - sinh(x-M[1])/(cosh(x-M[1]))^2 )
        elseif moduli == "maB"
            f = tanh(gamma*(x+M[1])) - tanh(gamma*(x-M[1])) - 1 + (M[2]/tanh(M[1]))*( sinh(gamma*(x+M[1]))/(cosh(gamma*(x+M[1])))^2 - sinh(gamma*(x-M[1]))/(cosh(gamma*(x-M[1])))^2 )
        elseif moduli == "pR"
            f = tanh(x+M[1]) - tanh(x-M[1]) - 1 + (M[2]/tanh(M[1]))*( (x+M[1])/cosh(x+M[1])^2 - (x-M[1])/cosh(x-M[1])^2 )
        elseif moduli == "mpR"
            f = tanh(gamma*(x+M[1])) - tanh(gamma*(x-M[1])) - 1 + (M[2]/tanh(M[1]))*( gamma*(x+M[1])/cosh(gamma*(x+M[1]))^2 - gamma*(x-M[1])/cosh(gamma*(x-M[1]))^2 )
        elseif moduli == "pR2"
	    f = ( (-(M[3]*(x+M[1])^2*tanh(x+M[1]))/(tanh(M[1])*cosh(x+M[1])^2))
		  +tanh(x+M[1])+(M[2]*(x+M[1]))/(tanh(M[1])*cosh(x+M[1])^2)
		  +(M[3]*(x-M[1])^2*tanh(x-M[1]))/(tanh(M[1])*cosh(x-M[1])^2)-tanh(x-M[1])
		  -(M[2]*(x-M[1]))/(tanh(M[1])*cosh(x-M[1])^2)-1
		)
	elseif moduli == "aBg"
	    f = tanh(M[3]*(x+M[1])) - tanh(M[3]*(x-M[1])) - 1 + (M[2]/tanh(M[1]))*( sinh(M[3]*(x+M[1]))/(cosh(M[3]*(x+M[1])))^2 - sinh(M[3]*(x-M[1]))/(cosh(M[3]*(x-M[1])))^2 )
	elseif moduli == "mpR2"
	    f = ( (M[3]*(x+M[1])^2*gamma^2*tanh((x+M[1])*gamma))/(tanh(M[1])*cosh((x+M[1])*gamma)^2)
		  +tanh((x+M[1])*gamma)-(M[2]*(x+M[1])*gamma)/(tanh(M[1])*cosh((x+M[1])*gamma)^2)
		  +(M[3]*(x-M[1])^2*gamma^2*tanh((x-M[1])*gamma))/(tanh(M[1])*cosh((x-M[1])*gamma)^2)
		  -tanh((x-M[1])*gamma)-(M[2]*(x-M[1])*gamma)/(tanh(M[1])*cosh((x-M[1])*gamma)^2)-1
		)

	end
    end
    return f
end

function U_kak(model,moduli, x, M,gamma)
    if model == "phi4"
        U = 0.5*(1-F_kak(model,moduli,x,M,gamma)^2)^2
    end
    return U
end

function W_kak(model,moduli,x, M,gamma)
    if model == "phi4"
        if moduli == "aB"
            deriv = -sech(M[1]-x)^2 + sech(M[1]+x)^2 + M[2]*coth(M[1])*(-sech(M[1]-x)^3 + sech(M[1]+x)^3 + sech(M[1]-x)*tanh(M[1]-x)^2 - sech(M[1]+x)*tanh(M[1]+x)^2 )
            W = 0.5*(deriv)^2 + U_kak(model,moduli,x,M,gamma)
        elseif moduli == "maB"
            deriv = -gamma*sech((-M[1]+x)*gamma)^2 + gamma*sech((M[1]+x)*gamma)^2 + M[2]*coth(M[1])*(-gamma*sech((-M[1]+x)*gamma)^3 + gamma*sech((M[1]+x)*gamma)^3 + gamma*sech((-M[1]+x)*gamma)*tanh((-M[1]+x)*gamma)^2 - gamma*sech((M[1]+x)*gamma)*tanh((M[1]+x)*gamma)^2 )
            W = 0.5*(deriv)^2 + U_kak(model,moduli,x,M,gamma)
        elseif moduli == "pR"
            deriv = -sech(M[1] - x)^2 + sech(M[1] + x)^2 + M[2]*coth(M[1])*(-sech(M[1] - x)^2 + sech(M[1] + x)^2 - 2*(-M[1] + x)*sech(M[1] - x)^2*tanh(M[1] - x) - 2*(M[1] + x)*sech(M[1] + x)^2*tanh(M[1] + x))
            W = 0.5*(deriv)^2 + U_kak(model,moduli,x,M,gamma)
        elseif moduli == "mpR"
            deriv = -gamma*sech((-M[1] + x)*gamma)^2 + gamma*sech((M[1] + x)*gamma)^2 + M[2]*coth(M[1])*(-gamma*sech((-M[1] + x)*gamma)^2 + gamma*sech((M[1] + x)*gamma)^2 + 2*(-M[1]+x)*gamma^2*sech((-M[1]+x)*gamma)^2*tanh((-M[1]+x)*gamma) - 2*(M[1]+x)*gamma^2*sech((M[1]+x)*gamma)^2*tanh((M[1]+x)*gamma))
            W = 0.5*(deriv)^2 + U_kak(model,moduli,x,M,gamma)
	elseif moduli == "pR2"
	    deriv = ( (2*M[3]*(x+M[1])^2*sinh(x+M[1])*tanh(x+M[1]))/(tanh(M[1])*cosh(x+M[1])^3)
		      -(2*M[3]*(x+M[1])*tanh(x+M[1]))/(tanh(M[1])*cosh(x+M[1])^2)
		      -(2*M[2]*(x+M[1])*sinh(x+M[1]))/(tanh(M[1])*cosh(x+M[1])^3)
		      -(M[3]*(x+M[1])^2*sech(x+M[1])^2)/(tanh(M[1])*cosh(x+M[1])^2)+sech(x+M[1])^2
		      +M[2]/(tanh(M[1])*cosh(x+M[1])^2)
		      -(2*M[3]*(x-M[1])^2*sinh(x-M[1])*tanh(x-M[1]))/(tanh(M[1])*cosh(x-M[1])^3)
		      +(2*M[3]*(x-M[1])*tanh(x-M[1]))/(tanh(M[1])*cosh(x-M[1])^2)
		      +(2*M[2]*(x-M[1])*sinh(x-M[1]))/(tanh(M[1])*cosh(x-M[1])^3)
		      +(M[3]*(x-M[1])^2*sech(x-M[1])^2)/(tanh(M[1])*cosh(x-M[1])^2)-sech(x-M[1])^2
		      -M[2]/(tanh(M[1])*cosh(x-M[1])^2) )
	    W = 0.5*(deriv)^2 + U_kak(model,moduli,x,M,gamma)
	elseif moduli == "aBg"
	    deriv = -M[3]*sech((-M[1]+x)*M[3])^2 + M[3]*sech((M[1]+x)*M[3])^2 + M[2]*coth(M[1])*(-M[3]*sech((-M[1]+x)*M[3])^3 + M[3]*sech((M[1]+x)*M[3])^3 + M[3]*sech((-M[1]+x)*M[3])*tanh((-M[1]+x)*M[3])^2 - M[3]*sech((M[1]+x)*M[3])*tanh((M[1]+x)*M[3])^2 )
            W = 0.5*(deriv)^2 + U_kak(model,moduli,x,M,gamma)
        elseif moduli == "mpR2"
	    deriv = ( (-(2*M[3]*(x+M[1])^2*gamma^3*sinh((x+M[1])*gamma)*tanh((x+M[1])*gamma))/(tanh(M[1])*cosh((x+M[1])*gamma)^3))
		      +(2*M[3]*(x+M[1])*gamma^2*tanh((x+M[1])*gamma))/(tanh(M[1])*cosh((x+M[1])*gamma)^2)
		      +(2*M[2]*(x+M[1])*gamma^2*sinh((x+M[1])*gamma))/(tanh(M[1])*cosh((x+M[1])*gamma)^3)
		      +(M[3]*(x+M[1])^2*gamma^3*sech((x+M[1])*gamma)^2)/(tanh(M[1])*cosh((x+M[1])*gamma)^2)+gamma*sech((x+M[1])*gamma)^2
		      -(M[2]*gamma)/(tanh(M[1])*cosh((x+M[1])*gamma)^2)
		      -(2*M[3]*(x-M[1])^2*gamma^3*sinh((x-M[1])*gamma)*tanh((x-M[1])*gamma))/(tanh(M[1])*cosh((x-M[1])*gamma)^3)
		      +(2*M[3]*(x-M[1])*gamma^2*tanh((x-M[1])*gamma))/(tanh(M[1])*cosh((x-M[1])*gamma)^2)
		      +(2*M[2]*(x-M[1])*gamma^2*sinh((x-M[1])*gamma))/(tanh(M[1])*cosh((x-M[1])*gamma)^3)
		      +(M[3]*(x-M[1])^2*gamma^3*sech((x-M[1])*gamma)^2)/(tanh(M[1])*cosh((x-M[1])*gamma)^2)-gamma*sech((x-M[1])*gamma)^2
		      -(M[2]*gamma)/(tanh(M[1])*cosh((x-M[1])*gamma)^2) )
	    W = 0.5*(deriv)^2 + U_kak(model,moduli,x,M,gamma)
	end
    end
    return W
end

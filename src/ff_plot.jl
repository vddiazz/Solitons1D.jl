using LinearAlgebra
using JLD2
using Plots
using LaTeXStrings

### colission animation

function ff_anim(ff_space,ff_time,path_to_file,savefig_bool)

    # savefig check
    if savefig_bool = true
        path_to_savefig = readline()

    # data loading
    @load path_to_file F Jarr Narr

    dt = Narr[end]/length(Narr)

    # plot animation
    for (t,tval) in enumerate(Narr)
        if n%20 == 0
            plt = plot(Jarr,F[:,t],
                       xlabel=L"x",
                       ylabel=L"\Phi_{\mathrm{KAK}}(x)",
                       xlim=(-10,10),
                       ylims=(-1.5,1.5),
                       title=L"v_{\mathrm{in}} = "*"$(-v0)"*L"; t = "*"$(round(t*dt,digits=3))",
                       legend=false,
                       color=:black)
            
            # show anim
            display(plt)
            
            # save figs
            if savefig_bool = true
                savefig(plt,path_to_savefig*"kak_t=$(lpad(t,5,'0')).png")

            sleep(0.01)
        end
    end
end

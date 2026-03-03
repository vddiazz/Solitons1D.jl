# Solitons1D

[![Build Status](https://github.com/vddiazz/Solitons1D.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/vddiazz/Solitons1D.jl/actions/workflows/CI.yml?query=branch%3Amain)

A Julia package to compute 1D soliton dynamics.

This package is being developed by *Víctor Díaz Díaz* as part of the Ph.D. thesis *Rigorous and analytical approximation methods in soliton theories*.

## Installation

This package is not registered in the Julia General Registry (yet).

To install this package, clone this repository:
```git
git clone https://github.com/vddiazz/Solitons1D.jl
```
Then move into the repository folder and run Julia. In the REPL do:
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Features

This package is aimed at computing kink-antikink (KAK) collisions using full-field (FF) dynamics and collective coordinate models (CCM).

### Full Field (FF):
- KAK collisions for $\phi^4$ theory on $\mathbb{R}$.
- KAK-KAK collisions for $\phi^4$ theory on $S^1$.

### Collective Coordinate Models (CCM):
- KAK moduli space dynamics for Poincaré mode + shape mode.
- KAK moduli space dynamics for Poincaré mode + modified shape mode.

## References

[1] Manton N., Sutcliffe P. (2004), *Topological Solitons*, Cambridge University Press.

[2] Press W.H., Teukolsky S.A., Vetterling W.T., Flannery B.P. (2007), *Numerical Recipes: The Art of Scientific Computing (3º ed.)*, Cambridge University Press.

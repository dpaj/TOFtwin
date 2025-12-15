cd("/maiqmag/vdp/TOFtwin")

using Pkg
Pkg.activate("/maiqmag/vdp/TOFtwin")
Pkg.add("StaticArrays")
Pkg.precompile()
using TOFtwin


#using .TOFtwin
pix = TOFtwin.pixels_from_coverage(L2=3.5, surface=:cylinder)
length(pix)

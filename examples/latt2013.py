from qcl import *
lattice = Q.Lattice([6,6,6,6])

U = lattice.GaugeField(2)
U.set_cold()
print U[(0,0,0,0),0]
print U[(0,0,0,0),0,0,0]

plaquette = (1,2,-1,-2)
print bc_symmetrize(plaquette)
print remove_duplicates(derive_paths(bc_symmetrize(plaquette),2))
rho = lattice.Field([1])
rho.set_link_product(U,[(1,2,3,4,-1,-2,-3,-4)])
print rho*rho
#print open('latest.c','r').read()

action = GaugeAction(U).add_term((1,2,-1,-2))
code = action.heatbath(beta=4.0,n_iter=100)
code.run()
print U.average_plaquette()

V = clone(U)
S = GaugeSmearOperator(U).add_term([1],1.0).add_term([2,1,-2],0.1)
V.set_smeared(S,reunitarize=4)
V.set_fat(U,'fat7+lepage')
V.set_hisq(U)

phi = lattice.FermiField(4,U.nc)
psi = clone(phi)
phi[(0,0,3,3),0,0] = 1.0

#U.set_cold()
kappa = 0.112
D = FermiOperator(U).add_diagonal_term(1.0)
for mu in (1,2,3,4):
    D.add_term(kappa*(I-G[mu]), [(mu,)])
    D.add_term(kappa*(I+G[mu]), [(-mu,)])
D.add_clover4d_terms(c_SW = 0.1)
psi.set(D,phi)
psi.set(invert_minimum_residue,D,phi)

#U.set_cold()
prop = lattice.ComplexScalarField()
for spin in range(4):
    for color in range(U.nc):
        psi = lattice.FermiField(4,U.nc)
        psi[(0,0,0,0),spin,color] = 1.0
        phi.set(invert_minimum_residue,D,psi)
        prop += make_meson(phi,I,phi)
prop_fft = prop.fft()
prop_t = [(t,math.log(prop_fft[t,0,0,0].real)) for t in range(lattice.shape[0])]
Canvas().plot(prop_t).save('meson.prop.png')

chi = psi.slice((0,0),(0,0))
Canvas().imshow(chi).save('fermi.propagator.png')
# print D.multiply(phi,psi).source

phi = lattice.StaggeredField(U.nc)
psi = clone(phi)
phi[(0,0,3,3),0] = 1.0

kappa = 0.112
D = FermiOperator(U).add_staggered_action(kappa = 0.112)
psi.set(D,phi)
psi.set(invert_minimum_residue,D,phi)

# check this works below!
chi = psi.slice((0,0),(0,))
Canvas().imshow(chi).save('staggered.propagator.png')

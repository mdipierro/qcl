from qcl import *
lattice = Q.Lattice([13,13,13,13])

U = lattice.GaugeField(2)
U.set_cold()

action = GaugeAction(U).add_term((1,2,-1,-2))
code = action.heatbath(beta=4.0,n_iter=100)
code.run()
print U.average_plaquette()

V = clone(U)
V.set_fat(U,'fat7',reunitarize=True)
U = V

kappa = 0.12
D = FermiOperator(U).add_diagonal_term(1.0)
for mu in (1,2,3,4):
    D.add_term(kappa*(I-G[mu]), [(+mu,)])
    D.add_term(kappa*(I+G[mu]), [(-mu,)])

prop = lattice.ComplexScalarField()
psi = lattice.FermiField(4,U.nc)
phi = clone(psi)
psi[(0,0,6,6),0,0] = 1.0
phi.set(invert_minimum_residue,D,psi)
chi = phi.slice((0,0),(0,0))
Canvas().imshow(chi).save('quark.propagator.0.0.therm.png')
chi = phi.slice((0,0),(1,0))
Canvas().imshow(chi).save('quark.propagator.1.0.therm.png')
chi = phi.slice((0,0),(2,0))
Canvas().imshow(chi).save('quark.propagator.2.0.therm.png')
chi = phi.slice((0,0),(3,0))
Canvas().imshow(chi).save('quark.propagator.3.0.therm.png')

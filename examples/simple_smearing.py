from qcl import *
lattice = Q.Lattice([13,13,13,13])
U = lattice.GaugeField(2)
U.set_cold()
phi = lattice.FermiField(4,U.nc)
psi = clone(phi)
phi[(0,0,6,6),0,0] = 1.0
S = FermiOperator(U).add_diagonal_term(1.0)
for mu in (1,2,3,4): 
    S.add_term(0.1*I,[(-mu,)]).add_term(0.1*I,[(mu,)])
for k in range(100):
    psi.set(S,phi)
    phi,psi = psi,phi
    chi = psi.slice((0,0),(0,0))
    Canvas().imshow(chi).save('smear.%.2i.png' % k)
    os.system('convert smear.%.2i.png smear.%.2i.jpg' % (k,k))

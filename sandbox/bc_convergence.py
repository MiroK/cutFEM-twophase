'''
This script investigates how setting boundary conditions effects convergence
rate of the numerical solution to Poisson problem with discontinuous diffusi-
vity.

Setting:
--------

    -div(kappa*grad(u)) = f in [-1, 1]^2
                    u   = g for x = -1 or x = 1
         kappa*grad(u)  = 0 for y = -1 or y = 1
      [kappa*grad(u)].n = 0 for x = 0
                  [u]   = 0 for x = 0

     kappa takes values k0 for x <=0 and k2 for x > 0

Methods:
--------
    1) The mesh is never aligned with the discontinuity and there is no special
    treatment of cells that have discontinuity of kappa inside

    2) Setting bcs strongly
    3) Setting bcs weakly with symmetric Nitsche formulation
    4) Setting bcs weakly with skew-Nitsche formulation
'''

from dolfin import *
from collections import defaultdict
from math import log as ln
import numpy as np

# Problem data ----------------------------------------------------------------
k0 = Constant(1.0)
k2 = Constant(100.0)
f = Expression('pi*pi*sin(pi*x[0])', degree=8)
u_exact = Expression('x[0]<=0 ? sin(pi*x[0])/k0 : sin(pi*x[0])/k2',
                     k0=k0, k2=k2)

# Domain definitions ----------------------------------------------------------

# Materials
subdomain0 = AutoSubDomain(lambda x: x[0] < DOLFIN_EPS)
subdomain2 = AutoSubDomain(lambda x: x[0] > 0)

# Boundaris
boundary0 = AutoSubDomain(lambda x, on_boundary: on_boundary and near(x[0], -1))
boundary1 = AutoSubDomain(lambda x, on_boundary: on_boundary and near((x[1]-1)*(x[1]+1), 0))
boundary2 = AutoSubDomain(lambda x, on_boundary: on_boundary and near(x[0], 1))


def create_domain(N):
    'Get mesh, meterials domains and boundaries.'
    mesh = RectangleMesh(-1, -1, 1, 1, N-1, N-1)

    material_domains = CellFunction('size_t', mesh, 0)
    subdomain0.mark(material_domains, 0)
    subdomain2.mark(material_domains, 2)

    boundaries = FacetFunction('size_t', mesh, 0)
    boundary0.mark(boundaries, 0)
    boundary1.mark(boundaries, 1)
    boundary2.mark(boundaries, 2)

    return mesh, material_domains, boundaries

# Solvers ---------------------------------------------------------------------

def strong(mesh, material_domains, boundaries):
    'Setting bcs strongly'
    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    dx = Measure('dx')[material_domains]
    a = inner(k0*grad(u), grad(v))*dx(0) + inner(k2*grad(u), grad(v))*dx(2)

    # The term over ds(1) is 0!
    L = inner(f, v)*dx('everywhere')

    bcs = [DirichletBC(V, u_exact, boundaries, 0),
           DirichletBC(V, u_exact, boundaries, 1)]
    A, b = assemble_system(a, L, bcs)

    uh = Function(V)
    solver = LUSolver()
    solver.solve(A, uh.vector(), b)

    return uh


def weak_sym(mesh, material_domains, boundaries):
    'Setting bcs via symmetric Nitsche'
    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    # Standard terms
    dx = Measure('dx')[material_domains]
    a = inner(k0*grad(u), grad(v))*dx(0) + inner(k2*grad(u), grad(v))*dx(2)

    ds = Measure('ds')[boundaries]
    n = FacetNormal(mesh)
    h = MaxFacetEdgeLength(mesh)
    gamma_h = 4
    # Nitsche terms
    a += - inner(k0*dot(grad(u), n), v)*ds(0)\
        - inner(k0*dot(grad(v), n), u)*ds(0)\
        + k0*gamma_h/h*inner(u, v)*ds(0)

    a += - inner(k2*dot(grad(u), n), v)*ds(2)\
        - inner(k2*dot(grad(v), n), u)*ds(2)\
        + k2*gamma_h/h*inner(u, v)*ds(2)

    # The term over ds(1) is 0!
    L = inner(f, v)*dx('everywhere')

    # Nitsche terms
    L += - inner(dot(k0*grad(v), n), u_exact)*ds(0)\
        + k0*gamma_h/h*inner(u_exact, v)*ds(0)

    # Nitsche terms for L in 2
    L += - inner(dot(k2*grad(v), n), u_exact)*ds(2)\
        + k2*gamma_h/h*inner(u_exact, v)*ds(2)

    A, b = assemble_system(a, L)

    uh = Function(V)
    solver = LUSolver()
    solver.solve(A, uh.vector(), b)

    return uh


def weak_skew(mesh, material_domains, boundaries):
    'Setting bcs via skew-symmetric Nitsche'
    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    # Standard terms
    dx = Measure('dx')[material_domains]
    a = inner(k0*grad(u), grad(v))*dx(0) + inner(k2*grad(u), grad(v))*dx(2)

    ds = Measure('ds')[boundaries]
    n = FacetNormal(mesh)
    # Nitsche terms
    a += - inner(k0*dot(grad(u), n), v)*ds(0)\
        + inner(k0*dot(grad(v), n), u)*ds(0)\

    a += - inner(k2*dot(grad(u), n), v)*ds(2)\
        + inner(k2*dot(grad(v), n), u)*ds(2)\

    # The term over ds(1) is 0!
    L = inner(f, v)*dx('everywhere')

    # Nitsche terms
    L += inner(dot(k0*grad(v), n), u_exact)*ds(0)\

    # Nitsche terms for L in 2
    L += inner(dot(k2*grad(v), n), u_exact)*ds(2)\

    A, b = assemble_system(a, L)

    uh = Function(V)
    solver = LUSolver()
    solver.solve(A, uh.vector(), b)

    return uh

# Test ------------------------------------------------------------------------

# Band for avoiding material discontinuity
band = AutoSubDomain(lambda x: -0.2 < x[0] < 0.2)

# Choose the solver <-----------------------------------------
solver = strong

# Gather data
data = defaultdict(list)
i_list = range(3, 8)
for i in i_list:
    N = 2**i
    mesh, material_domains, boundaries = create_domain(N)
    uh = solver(mesh, material_domains, boundaries)

    errorL2 = errornorm(u_exact, uh, 'L2')
    errorH10 = errornorm(u_exact, uh, 'H10')

    cell_f = CellFunction('size_t', mesh, 1)
    band.mark(cell_f, 0)
    submesh = SubMesh(mesh, cell_f, 1)

    errorL2_sub = errornorm(u_exact, uh, 'L2', mesh=submesh)
    errorH10_sub = errornorm(u_exact, uh, 'H10', mesh=submesh)

    data['h'].append(mesh.hmax())
    data['L2'].append(errorL2)
    data['H10'].append(errorH10)
    data['L2_sub'].append(errorL2_sub)
    data['H10_sub'].append(errorH10_sub)

# Compute rates
for i in range(1, len(i_list)):
    h, h_ = data['h'][i], data['h'][i-1]
    norms = []
    for norm in ['H10', 'H10_sub', 'L2', 'L2_sub']:
        e, e_ = data[norm][i], data[norm][i-1]
        rate = ln(e/e_)/ln(h/h_)
        norms.append('='.join([norm, '%.2f' % rate]))
    print '%.4E' % h, ' '.join(norms)

# With k0=k1 the rates are same for all the solver
# Once discontinuity is present, solver with strong bcs yields:
#  1.8856E-01 H10=0.58 H10_sub=0.98 L2=1.60 L2_sub=1.98
#  9.1240E-02 H10=0.55 H10_sub=1.01 L2=1.56 L2_sub=2.01
#  4.4896E-02 H10=0.53 H10_sub=1.00 L2=1.54 L2_sub=2.00
#  2.2271E-02 H10=0.51 H10_sub=1.01 L2=1.52 L2_sub=2.01

# While Nitsche symmetric solver yields:
#  1.8856E-01 H10=0.50 H10_sub=0.90 L2=0.84 L2_sub=0.49
#  9.1240E-02 H10=0.51 H10_sub=1.01 L2=0.92 L2_sub=1.12
#  4.4896E-02 H10=0.51 H10_sub=1.00 L2=0.96 L2_sub=0.97
#  2.2271E-02 H10=0.50 H10_sub=1.01 L2=0.98 L2_sub=1.03

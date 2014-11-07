from dolfin import *
from collections import defaultdict
from math import log as ln

# Problem data ----------------------------------------------------------------
f = Expression('pi*pi*sin(pi*x[0])', degree=8)
u_exact = Expression('sin(pi*x[0])', degree=8)

d_boundary = AutoSubDomain(lambda x, on_boundary:
                           on_boundary and near(x[0]*(1-x[0]), 0))

def strong(mesh, order):
    'Setting bcs strongly'
    V = FunctionSpace(mesh, 'CG', order)
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx
    boundaries = FacetFunction('size_t', mesh, 0)
    d_boundary.mark(boundaries, 1)
    bc = DirichletBC(V, u_exact, boundaries, 1)
    A, b = assemble_system(a, L, bc)

    uh = Function(V)
    solver = LUSolver()
    solver.solve(A, uh.vector(), b)

    return uh


def weak_sym(mesh, order):
    'Setting bcs via symmetric Nitsche'
    V = FunctionSpace(mesh, 'CG', order)
    u = TrialFunction(V)
    v = TestFunction(V)

    # Standard terms
    a = inner(grad(u), grad(v))*dx
    # Nitsche terms
    n = FacetNormal(mesh)
    h = MaxFacetEdgeLength(mesh)
    gamma_h = Constant(100)
    boundaries = FacetFunction('size_t', mesh, 0)
    d_boundary.mark(boundaries, 1)
    ds = Measure('ds')[boundaries]
    a += - inner(dot(grad(u), n), v)*ds(1) - inner(dot(grad(v), n), u)*ds(1)\
        + gamma_h/h*inner(u, v)*ds(1)

    L = inner(f, v)*dx('everywhere')
    # Nitsche terms
    L += - inner(dot(grad(v), n), u_exact)*ds(1) +\
        gamma_h/h*inner(u_exact, v)*ds(1)

    A, b = assemble_system(a, L)

    uh = Function(V)
    solver = LUSolver()
    solver.solve(A, uh.vector(), b)

    return uh


def weak_skew(mesh, order):
    'Setting bcs via skew-symmetric Nitsche'
    V = FunctionSpace(mesh, 'CG', order)
    u = TrialFunction(V)
    v = TestFunction(V)

    # Standard terms
    a = inner(grad(u), grad(v))*dx

    # Nitsche terms
    n = FacetNormal(mesh)
    boundaries = FacetFunction('size_t', mesh, 0)
    d_boundary.mark(boundaries, 1)
    ds = Measure('ds')[boundaries]
    a += - inner(dot(grad(u), n), v)*ds(1) + inner(dot(grad(v), n), u)*ds(1)

    L = inner(f, v)*dx('everywhere')

    # Nitsche terms
    L += inner(dot(grad(v), n), u_exact)*ds(1)

    A, b = assemble_system(a, L)

    uh = Function(V)
    solver = LUSolver()
    solver.solve(A, uh.vector(), b)

    return uh

# Test ------------------------------------------------------------------------

order = 2
for name, solver in {'strong': strong,
                     'weak_sym': weak_sym,
                     'weak_skew': weak_skew}.items():
    # Gather data
    data = defaultdict(list)
    i_list = range(3, 9)
    for i in i_list:
        N = 2**i
        mesh = UnitSquareMesh(N, N)
        uh = solver(mesh, order)

        errorL2 = errornorm(u_exact, uh, 'L2')
        errorH10 = errornorm(u_exact, uh, 'H10')

        # At some point the L2 error seizes to decrease

        data['h'].append(mesh.hmax())
        data['L2'].append(errorL2)
        data['H10'].append(errorH10)

    # Compute rates
    print '-'*79
    print '\t\t\t', name
    for i in range(1, len(i_list)):
        h, h_ = data['h'][i], data['h'][i-1]
        norms = []
        for norm in ['H10', 'L2']:
            e, e_ = data[norm][i], data[norm][i-1]
            rate = ln(e/e_)/ln(h/h_)
            norms.append('='.join([norm, '%.2f' % rate]))
        print '%.4E' % h, ' '.join(norms)

# The drop in order here is due to roundoff @ 10E-12 and is fixed by introducing
# a more challenging right hand side

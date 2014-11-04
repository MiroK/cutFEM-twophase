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
import subprocess
import os

# Problem data ----------------------------------------------------------------
k0 = Constant(1.0)
k2 = Constant(100.0)
f = Expression('pi*pi*sin(pi*x[0])', degree=8)
u_exact = Expression('x[0] < DOLFIN_EPS? sin(pi*x[0])/k0 : sin(pi*x[0])/k2',
                     k0=k0, k2=k2, degree=8)

# Domain definitions ----------------------------------------------------------

# Boundaries
boundary0 = AutoSubDomain(lambda x, on_boundary:
                          on_boundary and near(x[0]+1, DOLFIN_EPS))
boundary1 = AutoSubDomain(lambda x, on_boundary:
                          on_boundary and near((x[1]-1)*(x[1]+1), DOLFIN_EPS))
boundary2 = AutoSubDomain(lambda x, on_boundary:
                          on_boundary and near(x[0]-1, DOLFIN_EPS))


def create_domain(i, geo_file=None):
    'Get mesh, meterials domains and boundaries by modifyng geo file.'
    if geo_file is None:
        N = 2**i - 1
        mesh = RectangleMesh(-1, -1, 1, 1, N, N)
    else:
        name, ext = os.path.splitext(geo_file)
        assert ext == '.geo', 'Not geo file'

        ext = ext.strip('.')

        # See if mesh created before
        xml_file = '.'.join([name, str(i), 'xml'])
        if os.path.exists(xml_file):
            mesh = Mesh(xml_file)
        else:
            # Read the original file
            with open(geo_file, 'r') as input_file:
                content = input_file.readlines()

            new_content = []
            new_content[:] = content

            FOUND_SIZE = False

            new_geo_file = '.'.join([name, str(i), ext])
            size = 2./2**i
            for j, line in enumerate(new_content):
                if 'size' in line:
                    new_content[j] = 'size = %g;\n' % size
                    FOUND_SIZE = True
                    break

            if not FOUND_SIZE:
                print '.geo file has no lines with size'
                return -1

            # Write to temporaty file
            with open(new_geo_file, 'w') as output:
                output.write(''.join(new_content))

            # Mesh the geo file
            failed = subprocess.call(['gmsh', '-2', '%s' % new_geo_file])

            if failed:
                print 'gmsh meshing failed'
                return -1

            # Now convert the .msh file to xml
            msh_file = '.'.join([name, str(i), 'msh'])
            if os.path.exists(msh_file):
                failed = subprocess.call(['dolfin-convert', msh_file, xml_file])

                if failed:
                    print 'dolfin-convert failed'
                    return -1

                os.remove(new_geo_file)
                os.remove(msh_file)

                mesh = Mesh(xml_file)

    facet_f = FacetFunction('size_t', mesh, 3)
    boundary0.mark(facet_f, 0)
    boundary1.mark(facet_f, 1)
    boundary2.mark(facet_f, 2)

    ds = Measure('ds')[facet_f]
    assert assemble(1*ds(0, domain=mesh)) - 2 < 1E-10
    assert assemble(1*ds(1, domain=mesh)) - 4 < 1E-10
    assert assemble(1*ds(2, domain=mesh)) - 2 < 1E-10
    assert assemble(1*ds(3, domain=mesh)) < 1E-15

    cell_f = CellFunction('size_t', mesh, 3)
    for cell in cells(mesh):
        if cell.midpoint().x() < 0:
            cell_f[cell] = 0
        else:
            cell_f[cell] = 2

    dx = Measure('dx')[cell_f]
    assert assemble(1*dx(3, domain=mesh)) < 1E-15

    return mesh, cell_f, facet_f

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
           DirichletBC(V, u_exact, boundaries, 2)]
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
    gamma_h = Constant(100)
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
for name, solver in {'strong': strong,
                     'weak_sym': weak_sym,
                     'weak_skew': weak_skew}.items():
    # Gather data
    data = defaultdict(list)
    i_list = range(3, 9)
    for i in i_list:
        mesh, material_domains, boundaries = create_domain(i,
                                                           geo_file=None)
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

    # Plot error on the last mesh as DG0 function
    # Ve = FunctionSpace(mesh, 'DG', 4)
    # u = interpolate(u_exact, Ve)
    # u.vector().axpy(-1, interpolate(uh, Ve).vector())
    # u.vector()[:] *= u.vector()
    # u.vector()[:] -= u.vector().max()

    # E = FunctionSpace(mesh, 'DG', 1)
    # e = Function(E)
    # dofs_x = E.dofmap().tabulate_all_coordinates(mesh).reshape((E.dim(), 2))
    # e.vector().set_local(np.array([u(dof_x) for dof_x in dofs_x]))
    # plot(e, title='error')
    # plot(material_domains, title='domains')
    # plot(boundaries, title='bdries')
    # plot(cell_f, title='band')

    # Compute rates
    print '-'*79
    print '\t\t\t', name
    for i in range(1, len(i_list)):
        h, h_ = data['h'][i], data['h'][i-1]
        norms = []
        for norm in ['H10', 'H10_sub', 'L2', 'L2_sub']:
            e, e_ = data[norm][i], data[norm][i-1]
            rate = ln(e/e_)/ln(h/h_)
            norms.append('='.join([norm, '%.2f' % rate]))
        print '%.4E' % h, ' '.join(norms)
    # interactive()

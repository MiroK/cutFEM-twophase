#ifndef __BELGIAN_SOLVER_H
#define __BELGIAN_SOLVER_H

#include "PoissonSolver.h"
#include "PoissonP1.h"
#include "PoissonP2.h"
#include "PoissonP3.h"
#include <dolfin.h>
#include <cutfem.h>

using namespace dolfin;
using namespace cutfem;

class LevelSet : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    //values[0] = x[0]; // vertical line
    values[0] = (x[1]-x[0])/sqrt(2); // slanted line
  }
};


class BelgianSolver : public PoissonSolver
{
public:
  // Set boundary conditions strongly
  std::pair<double, double>
    solve(const std::shared_ptr<Mesh> mesh,
          const FacetFunction<size_t>& boundaries,
          const Expression& f,
          const Expression& g,
          const std::size_t order=2) const;
};

// ----------------------------------------------------------------------------

std::pair<double, double>
BelgianSolver::solve(const std::shared_ptr<Mesh> mesh,
                     const FacetFunction<size_t>& boundaries,
                     const Expression& f,
                     const Expression& g,
                     const std::size_t order) const
{
  Constant gamma_h(20);
  Constant k0(1), k2(10);
  std::size_t quad_order = 2;

  // Actual solver
  CompositeMesh composite_mesh;
  composite_mesh.add(mesh);
  // Empty functions for now.
  composite_mesh.compute_collisions();
  composite_mesh.compute_intersections();

  // Compute level set discretization
  std::shared_ptr<Expression> phi(new LevelSet);
  MeshCutter mesh_cutter(mesh, phi);
  mesh_cutter.compute_cut_mesh_topology();
  mesh_cutter.compute_cut_mesh_geometry();

  // Create function spaces on each mesh and composite function space
  PoissonP2::FunctionSpace V0(mesh);

  // Create a simple composite function space consisting of single
  // a function space.
  CompositeFunctionSpace V(composite_mesh);
  V.add(V0);
  V.build();

  // Create empty CompositeForm
  CompositeForm a(V, V);

  // Create standard bilinear forms on mesh 0
  PoissonP2::Form_a_belg a0(V0, V0);
  a0.set_cell_domains(mesh_cutter.domain_marker());
  a0.ds = boundaries;
  a0.gamma_h = gamma_h;
  a0.k0 = k0;
  a0.k2 = k2;

  // Add them as CutForm to CompositeForm
  a.add(std::shared_ptr<CutForm>(new CutForm(a0)));

  // Extract inner cut cells.
  auto cut_cells_0 = mesh_cutter.cut_cells()[0];
  // Extract inner cut cells. Note we have 2 vs 1.
  auto cut_cells_2 = mesh_cutter.cut_cells()[1];

  // All cut meshes have the one and only mesh 0 as parent
  std::vector<std::size_t> parent_mesh_ids;
  parent_mesh_ids.push_back(0);

  // Define quadrature rules for the cut meshes
  std::shared_ptr<Quadrature> quadrature_cut_cells_0
  (new Quadrature(cut_cells_0->type().cell_type(),
      cut_cells_0->geometry().dim(),
      quad_order));

  std::shared_ptr<Quadrature> quadrature_cut_cells_2
  (new Quadrature(cut_cells_2->type().cell_type(),
      cut_cells_2->geometry().dim(),
      quad_order));

  a.cut_form(0)->set_quadrature(0, quadrature_cut_cells_0);
  a.cut_form(0)->set_cut_mesh(0, cut_cells_0);
  a.cut_form(0)->set_parent_mesh_ids(0, parent_mesh_ids);

  //Provide same information for domain 2 in form 0
  a.cut_form(0)->set_quadrature(2, quadrature_cut_cells_2);
  a.cut_form(0)->set_cut_mesh(2, cut_cells_2);
  a.cut_form(0)->set_parent_mesh_ids(2, parent_mesh_ids);

  // Create forms for rhs
  CompositeForm L(V);
  PoissonP2::Form_L_belg L0(V0);
  L0.f = f;
  L0.g = g;
  L0.ds = boundaries;
  L0.gamma_h = gamma_h;
  L0.k0 = k0;
  L0.k2 = k2;
  L.add(std::shared_ptr<CutForm>(new CutForm(L0)));

  // Assemble system
  Matrix A;
  Vector b;

  CompositeFormAssembler assembler;
  assembler.assemble(A, a);
  assembler.assemble(b, L);

  // Solve
  CompositeFunction uh(V);
  PETScLUSolver solver;
  solver.solve(A, *uh.vector(), b);

  Function solution(uh.part(0));

  return error_norm(g, solution);
}

#endif // __BELGIAN_SOLVER_H

#ifndef __STRONG_SOLVER_H
#define __STRONG_SOLVER_H

#include "PoissonSolver.h"
#include "PoissonP1.h"
#include "PoissonP2.h"
#include "PoissonP3.h"
#include <dolfin.h>

using namespace dolfin;

class StrongSolver : public PoissonSolver
{
public:
  // Set boundary conditions strongly
  std::pair<double, double>
    solve(const std::shared_ptr<Mesh> mesh,
          const FacetFunction<size_t>& boundaries,
          const Expression& f,
          const Expression& g,
          const std::size_t order) const;
};

// ----------------------------------------------------------------------------

std::pair<double, double>
StrongSolver::solve(const std::shared_ptr<Mesh> mesh,
                    const FacetFunction<size_t>& boundaries,
                    const Expression& f,
                    const Expression& g,
                    const std::size_t order) const
{
  if(order == 1)
  {
    PoissonP1::FunctionSpace V(mesh);
    PoissonP1::Form_a_strong a(V, V);
    PoissonP1::Form_L_strong L(V);
    L.f = f;

    DirichletBC bc(V, g, boundaries, 1);
    Function uh(V);
    dolfin::solve(a == L, uh, bc);

    return error_norm(g, uh);
  }
  if(order == 2)
  {
    PoissonP2::FunctionSpace V(mesh);
    PoissonP2::Form_a_strong a(V, V);
    PoissonP2::Form_L_strong L(V);
    L.f = f;

    DirichletBC bc(V, g, boundaries, 1);
    Function uh(V);
    dolfin::solve(a == L, uh, bc);

    return error_norm(g, uh);
  }
  if(order == 3)
  {
    PoissonP3::FunctionSpace V(mesh);
    PoissonP3::Form_a_strong a(V, V);
    PoissonP3::Form_L_strong L(V);
    L.f = f;

    DirichletBC bc(V, g, boundaries, 1);
    Function uh(V);
    dolfin::solve(a == L, uh, bc);

    return error_norm(g, uh);
  }
}


#endif // __STRONG_SOLVER_H

#ifndef __WEAK_SYM_SOLVER_H
#define __WEAK_SYM_SOLVER_H

#include "PoissonSolver.h"
#include "PoissonP1.h"
#include "PoissonP2.h"
#include "PoissonP3.h"
#include <dolfin.h>

using namespace dolfin;

class WeakSymSolver : public PoissonSolver
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
WeakSymSolver::solve(const std::shared_ptr<Mesh> mesh,
                     const FacetFunction<size_t>& boundaries,
                     const Expression& f,
                     const Expression& g,
                     const std::size_t order) const
{
  if(order == 1)
  {
    Constant gamma_h(10);

    PoissonP1::FunctionSpace V(mesh);
    PoissonP1::Form_a_weak_sym a(V, V);
    a.ds = boundaries;
    a.gamma_h = gamma_h;

    PoissonP1::Form_L_weak_sym L(V);
    L.ds = boundaries;
    L.gamma_h = gamma_h;
    L.f = f;
    L.g = g;

    Function uh(V);
    dolfin::solve(a == L, uh);

    return error_norm(g, uh);
  }
  if(order == 2)
  {
    Constant gamma_h(20);

    PoissonP2::FunctionSpace V(mesh);
    PoissonP2::Form_a_weak_sym a(V, V);
    a.ds = boundaries;
    a.gamma_h = gamma_h;

    PoissonP2::Form_L_weak_sym L(V);
    L.ds = boundaries;
    L.gamma_h = gamma_h;
    L.f = f;
    L.g = g;

    Function uh(V);
    dolfin::solve(a == L, uh);

    return error_norm(g, uh);
  }
  if(order == 3)
  {
    Constant gamma_h(40);

    PoissonP3::FunctionSpace V(mesh);
    PoissonP3::Form_a_weak_sym a(V, V);
    a.ds = boundaries;
    a.gamma_h = gamma_h;

    PoissonP3::Form_L_weak_sym L(V);
    L.ds = boundaries;
    L.gamma_h = gamma_h;
    L.f = f;
    L.g = g;

    Function uh(V);
    dolfin::solve(a == L, uh);

    return error_norm(g, uh);
  }
}

#endif // __WEAK_SYM_SOLVER_H

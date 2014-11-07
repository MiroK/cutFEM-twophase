#ifndef __POISSON_SOLVER_H
#define __POISSON_SOLVER_H

#include <dolfin.h>
#include "Error.h"

using namespace dolfin;

class PoissonSolver
{
public:
  // Solve the Poisson with given source f on a mesh with dirichlet boundary
  // value g(exact solution) on boundaries using CG element of order and
  // return H10 and L2 norms of error
  virtual std::pair<double, double>
    solve(const std::shared_ptr<Mesh> mesh,
          const FacetFunction<size_t>& boundaries,
          const Expression& f,
          const Expression& g,
          const std::size_t order) const = 0;

  // Compute H10 and L2 error
 std::pair<double, double> error_norm(const Expression& u, const Function& uh) const;
};

// ----------------------------------------------------------------------------

std::pair<double, double>
PoissonSolver::error_norm(const Expression& u, const Function& uh) const
{
  // interpolate u and uh onto error space
  std::shared_ptr<const  Mesh> mesh(uh.function_space()->mesh());
  Error::CoefficientSpace_u Ve(mesh);
  Function u_e(Ve), uh_e(Ve);
  u_e.interpolate(u);
  uh_e.interpolate(uh);

  // assemble the H10 error form to get H10 error
  Error::Form_h10_norm form_h10(mesh, u_e, uh_e);
  double h10_error = sqrt(assemble(form_h10));
  
  // asssemble the L2 error form to get L2 error
  Error::Form_l2_norm form_l2(mesh, u_e, uh_e);
  double l2_error = sqrt(assemble(form_l2));

  std::pair<double, double> errors(h10_error, l2_error);
  return errors;
}

#endif // __POINSSON_SOLVER_H

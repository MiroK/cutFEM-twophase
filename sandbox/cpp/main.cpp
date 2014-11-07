#include "PoissonSolver.h"
#include "StrongSolver.h"
#include "WeakSymSolver.h"
#include "WeakSkewSolver.h"
#include "BelgianSolver.h"

using namespace dolfin;

// source f
class Source : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    //values[0] = M_PI*M_PI*sin(M_PI*x[0]);
    double ans = -2*M_PI*x[0]*x[0]*sin(4*M_PI*x[0])*sin(M_PI*x[0]*x[0]);
    ans += 8*M_PI*x[0]*cos(4*M_PI*x[0])*cos(M_PI*x[0]*x[0]);
    ans += -8*M_PI*sin(4*M_PI*x[0])*sin(M_PI*x[0]*x[0]);
    ans += sin(4*M_PI*x[0])*cos(M_PI*x[0]*x[0]);
    values[0] = 2*M_PI*ans;
  }
};

// exact solution
class ExactSolution : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    //values[0] = sin(M_PI*x[0]);
    values[0] = -sin(4*M_PI*x[0])*sin(M_PI*x[0]*x[0]);
  }
};

// domain for dirichlet boundary conditions
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS;
  }
};

// ----------------------------------------------------------------------------

int main()
{
  set_log_level(WARNING);

  Source f;
  ExactSolution u;
  DirichletBoundary dirichlet_boundary;

  StrongSolver solver0;
  WeakSymSolver solver1;
  WeakSkewSolver solver2;
  BelgianSolver solver3;
  std::map<std::string, PoissonSolver*> solvers;
  solvers["strong"] = &solver0;
  solvers["weak_sym"] = &solver1;
  solvers["weak_skew"] = &solver2;
  solvers["belgian"] = &solver3;

  std::size_t order = 3;
  for(auto& solver : solvers)
  {
    std::cout << solver.first << std::endl;
    double h10_, l2_, h_;
    for(std::size_t i = 3; i < 9 - (order > 2 ? 1 : 0); i++)
    {
      std::size_t N = pow(2, i);
      std::shared_ptr<Mesh> mesh(new UnitSquareMesh(N, N));

      FacetFunction<size_t> boundaries(*mesh, 4);
      dirichlet_boundary.mark(boundaries, 1);

      std::pair<double, double>
        errors = solver.second->solve(mesh, boundaries, f, u, order);
      double h10 = errors.first, l2 = errors.second, h = mesh->hmax();

      if(i > 3)
      {
        double h10_rate = std::log10(h10/h10_)/std::log10(h/h_);
        double l2_rate = std::log10(l2/l2_)/std::log10(h/h_);

        warning("h = %.4E | e_h10 = %.4E, r_h10 = %.2f | e_l2 = %.4E, r_l2 = %.2f",
            h, h10, h10_rate, l2, l2_rate);

      }
      h10_ = h10;
      l2_ = l2;
      h_ = h;
    }
  }
  return 0;
}

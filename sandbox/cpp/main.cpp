#include "PoissonSolver.h"
#include "StrongSolver.h"
#include "WeakSymSolver.h"
#include "WeakSkewSolver.h"
#include "BelgianSolver.h"
#include "CutFemSolver.h"

using namespace dolfin;

// source f
class Source : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    //values[0] = M_PI*M_PI*sin(M_PI*x[0]);  // vertical line
    values[0] = M_PI*M_PI*sin(M_PI*(x[1]-x[0])/sqrt(2));  // slanted line
  }
};

// exact solution
class ExactSolution : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    double k0 = 1, k2 = 10;
    //values[0] = x[0] <= 0 ? sin(M_PI*x[0])/k0 : sin(M_PI*x[0])/k2; 
    values[0] = x[1] <= x[0] ? sin(M_PI*(x[1]-x[0])/sqrt(2))/k0:
                               sin(M_PI*(x[1]-x[0])/sqrt(2))/k2;
  }
};

// Left GammaD
class DirichletBoundaryLeft : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return x[0] < -1 + DOLFIN_EPS;
  }
};


// Right Gamma_D
class DirichletBoundaryRight : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return x[0] > 1 - DOLFIN_EPS;
  }
};

// ----------------------------------------------------------------------------

int main()
{
  set_log_level(WARNING);

  Source f;
  ExactSolution u;
  DirichletBoundaryLeft left_boundary;
  DirichletBoundaryRight right_boundary;

  BelgianSolver solver0;
  CutFemSolver solver1;
  std::map<std::string, PoissonSolver*> solvers;
  //solvers["belgian"] = &solver0;
  solvers["cutfem"] = &solver1;

  for(auto& solver : solvers)
  {
    std::cout << solver.first << std::endl;
    double h10_, l2_, h_;
    for(std::size_t i = 3; i < 9; i++)
    {
      std::size_t N = pow(2, i);
      std::shared_ptr<Mesh> mesh(new RectangleMesh(-1, -1, 1, 1, N-1, N));

      FacetFunction<size_t> boundaries(*mesh, 4);

      // rotate for slanted line  
      left_boundary.mark(boundaries, 3);
      right_boundary.mark(boundaries, 1);
      Point origin(0., 0.);
      mesh->rotate(315, 2, origin);
      
      std::pair<double, double>
        errors = solver.second->solve(mesh, boundaries, f, u, 1);
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
  interactive();
  return 0;
}

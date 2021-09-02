#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "thtrdat_.hpp"
#include "compute.hpp"
#include <math.h>
#include <numeric>

void transport(block_ b,
             thtrdat_ th,
                 int  face,
         std::string  given) {

  MDRange3 range = get_range3(b, face);


  Kokkos::parallel_for("Compute transport properties mu,kappa,Dij from poly'l",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

  // poly'l order
  const int po = 4;
  int ns=th.ns;
  double p;
  double T;
  double Y[ns],X[ns];

  double Rmix,MWmix;

  p = b.q(i,j,k,0);
  T = b.q(i,j,k,4);
  // Compute nth species Y
  Y[ns-1] = 1.0;
  for (int n=0; n<ns-1; n++)
  {
    Y[n] = b.q(i,j,k,5+n);
    Y[ns-1] -= Y[n];
  }
  Y[ns-1] = std::max(0.0,Y[ns-1]);

  // Update mixture properties
  Rmix = 0.0;
  MWmix = 0.0;
  for (int n=0; n<=ns-1; n++)
  {
    Rmix  += Y[n]*th.Ru/th.MW[n];
    MWmix += Y[n]*th.MW[n];
  }
  // Mole fractions
  for (int n=0; n<=ns-1; n++)
  {
    X[n] = Y[n]/th.MW[i]*MWmix;
  }

  // Evaluate all property polynomials
  double MW = th.MW;
  //viscosity
  double mu_sp[ns] = 0.0;
  double phi[ns][ns] = 0.0;
  //thermal conductivity
  double kappa_sp[ns] = 0.0;
  // binary diffusion
  double Dij[ns][ns] = 0.0;

  std::vector

  for (int n=0; n<=ns-1; n++)
  {
    //Set it to constant value first
    mu_sp[n] = th.mu_p[n][po]
    kappa_sp[n] = th.kappa_p[n][po]
    for (int p=0; p<po; p++)
    {
      mu_sp[n] += th.mu_p[n][p]*pow(T,float(po-p))
      kappa_sp[n] += th.kappa_p[n][p]*pow(T,float(po-p))
    }
  }

  });
}

#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "thtrdat_.hpp"
#include "compute.hpp"
#include <math.h>
#include <numeric>

void constantProps(block_ b,
             const thtrdat_ th,
             const int face,
             const int i/*=0*/,
             const int j/*=0*/,
             const int k/*=0*/) {

  MDRange3 range = get_range3(b, face, i, j, k);

  Kokkos::parallel_for("Compute transport properties mu,kappa,Dij from poly'l",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

  int ns=th.ns;
  double Y[ns],X[ns];

  double Rmix,MWmix;

  // Compute nth species Y
  Y[ns-1] = 1.0;
  for (int n=0; n<ns-1; n++)
  {
    Y[n] = b.q(i,j,k,5+n);
    Y[ns-1] -= Y[n];
  }
  Y[ns-1] = std::max(0.0,Y[ns-1]);

  // Update mixture properties
  // Mole fractions
  double mass=0.0;
  for (int n=0; n<=ns-1; n++)
  {
    mass += Y[n]/th.MW[n];
  }
  for (int n=0; n<=ns-1; n++)
  {
    X[n] = Y[n]/th.MW[n]/mass;
  }
  // Mean molecular weight
  MWmix = 0.0;
  for (int n=0; n<=ns-1; n++)
  {
    MWmix += X[n]*th.MW[n];
  }

  // viscosity mixture
  double phi[ns][ns];
  for (int n=0; n<=ns-1; n++)
  {
    for (int n2=0; n2<=ns-1; n2++)
    {
      phi[n][n2] =  pow((1.0 + sqrt(th.mu0[n]/th.mu0[n2]*sqrt(th.MW[n2]/th.MW[n]))),2.0) /
                       ( sqrt(8.0)*sqrt(1+th.MW[n]/th.MW[n2]));
    }
  }

  double mu = 0.0;
  double phitemp;
  for (int n=0; n<=ns-1; n++)
  {
    phitemp = 0.0;
    for (int n2=0; n2<=ns-1; n2++)
    {
      phitemp += phi[n][n2]*X[n2];
    }
    mu += th.mu0[n]*X[n]/phitemp;
  }

  // thermal conductivity mixture
  double kappa = 0.0;

  double sum1=0.0;
  double sum2=0.0;
  for (int n=0; n<=ns-1; n++)
  {
    sum1 += X[n] * th.kappa0[n];
    sum2 += X[n] / th.kappa0[n];
  }
  kappa = 0.5*(sum1+1.0/sum2);

  // Set values of new properties
  // viscocity
  b.qt(i,j,k,0) = mu;
  // thermal conductivity
  b.qt(i,j,k,1) = kappa;
  // Diffusion coefficients mass
  // NOTE: Unity Lewis number approximation!
  for (int n=0; n<=ns-1; n++)
  {
    b.qt(i,j,k,2+n) = kappa / ( b.Q(i,j,k,0) * b.qh(i,j,k,1) );
  }

  });
}

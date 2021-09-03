#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "thtrdat_.hpp"
#include "compute.hpp"
#include <math.h>
#include <numeric>

void transport(block_ b,
             thtrdat_ th,
                 int  face) {

  MDRange3 range = get_range3(b, face);


  Kokkos::parallel_for("Compute transport properties mu,kappa,Dij from poly'l",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

  // poly'l order
  const int plyo = 4;
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
  MWmix = 0.0;
  for (int n=0; n<=ns-1; n++)
  {
    MWmix += Y[n]*th.MW[n];
  }
  // Mole fractions
  for (int n=0; n<=ns-1; n++)
  {
    X[n] = Y[n]/th.MW[i]*MWmix;
  }

  // Evaluate all property polynomials

  //viscosity
  double mu_sp[ns] = {0.0};
  //thermal conductivity
  double kappa_sp[ns] = {0.0};
  // binary diffusion
  double Dij[ns][ns] = {0.0};

  int indx;
  for (int n=0; n<=ns-1; n++)
  {
    //Set to constant value first
    mu_sp[n] = th.mu_poly[n][plyo];
    kappa_sp[n] = th.kappa_poly[n][plyo];

    for (int n2=n; n2<=ns-1; n2++)
    {
      indx = int(ns*(ns-1)/2 - (n2-n)*(ns-n-1)/2 + n2);
      Dij[n ][n2] = th.Dij_poly[indx][plyo];
      Dij[n2][n ] = Dij[n ][n2];
    }

    // Evaluate polynomial
    for (int ply=0; ply<plyo; ply++)
    {
      mu_sp[n] += th.mu_poly[n][ply]*pow(T,float(plyo-ply));
      kappa_sp[n] += th.kappa_poly[n][ply]*pow(T,float(plyo-ply));

      for (int n2=n; n2<=ns-1; n2++)
      {
        indx = int(ns*(ns-1)/2 - (n2-n)*(ns-n-1)/2 + n2);
        Dij[n ][n2] += th.Dij_poly[indx][ply]*pow(T,float(plyo-ply));
        Dij[n2][n ]  = Dij[n ][n2];
      }
    }
  }

  for (int n=0; n<=ns-1; n++)
  {
    std::cout << n << " " << kappa_sp[n] << "\n";
  }
  // Now every species' property is computed, generate mixture values

  // viscosity mixture
  double mu = 0.0;
  double phi[ns][ns] = {0.0};
  for (int n=0; n<=ns-1; n++)
  {
    for (int n2=0; n2<=ns-1; n2++)
    {
      phi[n][n2] =  pow((1.0 + sqrt(mu_sp[n]/mu_sp[n2]*sqrt(th.MW[n2]/th.MW[n]))),2.0) /
                       ( sqrt(8.0)*sqrt(1+th.MW[n]/th.MW[n2]));
    }
    double phitemp = 0.0;
    for (int n2=0; n2<=ns-1; n2++)
    {
      phitemp += phi[n][n2]*X[n2];
    }
    mu += mu_sp[n]*X[n]/phitemp;
  }

  // thermal conductivity mixture
  double kappa = 0.0;

  double sum1=0.0;
  double sum2=0.0;
  for (int n=0; n<=ns-1; n++)
  {
    sum1 += X[n] * kappa_sp[n];
    sum2 += X[n] / kappa_sp[n];
  }
  kappa = 0.5*(sum1+1.0/sum2);


  // mass diffusion coefficient mixture
  double D[ns] = {0.0};
  for (int n=0; n<=ns-1; n++)
  {
    sum1 = 0.0;
    sum2 = 0.0;
    for (int n2=0; n2<=ns-1; n2++)
    {
      if ( n == n2 )
      {
        continue;
      }
      sum1 += X[n2] / Dij[n][n2];
      sum2 += X[n2] * th.MW[n2] / Dij[n][n2];
    }
    sum1 *= p;
    sum2 *= p * X[n] / ( MWmix - th.MW[n]*X[n] );
    D[n] = 1.0 / (sum1 + sum2);
  }
  // Set values of new properties
  // viscocity
  b.qt(i,j,k,0) = mu;
  // thermal conductivity
  b.qt(i,j,k,1) = kappa;
  // Diffusion coefficients mass
  for (int n=0; n<ns-1; n++)
  {
    b.qt(i,j,k,2+n) = D[n];
  }

  });
}

#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "thtrdat_.hpp"
#include "compute.hpp"
#include <math.h>

void kineticTheoryUnityLewis(block_ b,
             const thtrdat_ th,
             const int nface,
             const int indxI/*=0*/,
             const int indxJ/*=0*/,
             const int indxK/*=0*/) {

  MDRange3 range = get_range3(b, nface, indxI, indxJ, indxK);
  Kokkos::Experimental::UniqueToken<exec_space> token;
  int numIds = token.size();

  const int ns=th.ns;
  twoDview Y("Y", ns, numIds);
  twoDview X("X", ns, numIds);
  //viscosity
  twoDview mu_sp("mu_sp", ns, numIds);
  //thermal conductivity
  twoDview kappa_sp("kappa_sp", ns, numIds);
  // binary diffusion
  twoDview D("D", ns, numIds);

  // poly'l degree
  const int deg = 4;

  Kokkos::parallel_for("Compute transport properties mu,kappa, from poly'l. Dij from unity Lewist assumption.",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {
  int id = token.acquire();

  double& T = b.q(i,j,k,4);


  // Compute nth species Y
  Y(ns-1,id) = 1.0;
  for (int n=0; n<ns-1; n++)
  {
    Y(n,id) = b.q(i,j,k,5+n);
    Y(ns-1,id) -= Y(n,id);
  }
  Y(ns-1,id) = fmax(0.0,Y(ns-1,id));

  // Update mixture properties
  // Mole fractions
  double mass=0.0;
  for (int n=0; n<=ns-1; n++)
  {
    mass += Y(n,id)/th.MW(n);
  }

  // Mean molecular weight, mole fraction
  double MWmix;
  MWmix = 0.0;
  for (int n=0; n<=ns-1; n++)
  {
    X(n,id) = Y(n,id)/th.MW(n)/mass;
    MWmix += X(n,id)*th.MW(n);
  }

  // Evaluate all property polynomials
  double logT = log(T);
  double sqrt_T = exp(0.5*logT);

  for (int n=0; n<=ns-1; n++)
  {
    //Set to constant value first
    mu_sp(n,id) = th.muPoly(n,deg);
    kappa_sp(n,id) = th.kappaPoly(n,deg);

    // Evaluate polynomial
    for (int ply=0; ply<deg; ply++)
    {
      mu_sp(n,id)    += th.muPoly(   n,ply)*pow(logT,float(deg-ply));
      kappa_sp(n,id) += th.kappaPoly(n,ply)*pow(logT,float(deg-ply));

    }

    // Set to the correct dimensions
    mu_sp(n,id) = sqrt_T*mu_sp(n,id);
    kappa_sp(n,id) = sqrt_T*kappa_sp(n,id);
  }

  // Now every species' property is computed, generate mixture values

  // viscosity mixture
  double phi;
  double mu = 0.0;
  for (int n=0; n<=ns-1; n++)
  {
    double phitemp = 0.0;
    for (int n2=0; n2<=ns-1; n2++)
    {
      phi =  pow((1.0 + sqrt(mu_sp(n,id)/mu_sp(n2,id)*sqrt(th.MW(n2)/th.MW(n)))),2.0) /
                       ( sqrt(8.0)*sqrt(1+th.MW(n)/th.MW(n2)));
      phitemp += phi*X(n2,id);
    }
    mu += mu_sp(n,id)*X(n,id)/phitemp;
  }

  // thermal conductivity mixture
  double kappa = 0.0;

  double sum1=0.0;
  double sum2=0.0;
  for (int n=0; n<=ns-1; n++)
  {
    sum1 += X(n,id) * kappa_sp(n,id);
    sum2 += X(n,id) / kappa_sp(n,id);
  }
  kappa = 0.5*(sum1+1.0/sum2);

  // Set values of new properties
  // viscocity
  b.qt(i,j,k,0) = mu;
  // thermal conductivity
  b.qt(i,j,k,1) = kappa;
  // NOTE: Unity Lewis number approximation!
  for (int n=0; n<=ns-1; n++)
  {
    b.qt(i,j,k,2+n) = kappa / ( b.Q(i,j,k,0) * b.qh(i,j,k,1) );
  }

  token.release(id);
  });
}

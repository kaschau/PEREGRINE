#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "thtrdat_.hpp"
#include "compute.hpp"
#include <math.h>
#include <stdexcept>

void tpg(block_ b,
   const thtrdat_ th,
   const int nface,
   const std::string given,
   const int indxI/*=0*/,
   const int indxJ/*=0*/,
   const int indxK/*=0*/) {

// For performance purposes, we want to compile with ns known whenever possible
// however, for testing, developement, etc. we want the flexibility to
// have it at run time as well. So we define some macros here to allow that.
#ifndef NSCOMPILE
  Kokkos::Experimental::UniqueToken<exec_space> token;
  int numIds = token.size();
  const int ns=th.ns;
  twoDview Y("Y", ns, numIds);
  twoDview hi("hi", ns, numIds);
#endif

#ifdef NSCOMPILE
  #define Y(INDEX) Y[INDEX]
  #define hi(INDEX) hi[INDEX]
  #define ns NS
#else
  #define Y(INDEX) Y(INDEX,id)
  #define hi(INDEX) hi(INDEX,id)
#endif

  MDRange3 range = get_range3(b, nface, indxI, indxJ, indxK);
  if ( given.compare("prims") == 0 )
  {
  Kokkos::parallel_for("Compute all conserved quantities from primatives via tgp",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {
#ifndef NSCOMPILE
  int id = token.acquire();
#endif

  // Updates all conserved quantities from primatives
  // Along the way, we need to compute mixture properties
  // gamma, cp, h, e, hi
  // So we store these as well.

  double& p = b.q(i,j,k,0);
  double& u = b.q(i,j,k,1);
  double& v = b.q(i,j,k,2);
  double& w = b.q(i,j,k,3);
  double& T = b.q(i,j,k,4);
#ifdef NSCOMPILE
  double Y(ns);
#endif

  double rho;
  double rhou,rhov,rhow;
  double e,tke,rhoE;
  double gamma,cp,h,c;
#ifdef NSCOMPILE
  double hi(ns);
#endif
  double Rmix;

  // Compute nth species Y
  Y(ns-1) = 1.0;
  for (int n=0; n<ns-1; n++)
  {
    Y(n) = b.q(i,j,k,5+n);
    Y(ns-1) -= Y(n);
  }
  Y(ns-1) = fmax(0.0,Y(ns-1));

  // Compute Rmix
  Rmix = 0.0;
  for (int n=0; n<=ns-1; n++)
  {
    Rmix += Y(n)/th.MW(n);
  }
  Rmix *= th.Ru;

  // Update mixture properties
  h    = 0.0;
  cp   = 0.0;
  for (int n=0; n<=ns-1; n++)
  {
    int m = ( T <= th.NASA7(n,0) ) ? 8 : 1;

    double cps=(th.NASA7(n,m+0)            +
                th.NASA7(n,m+1)*    T      +
                th.NASA7(n,m+2)*pow(T,2.0) +
                th.NASA7(n,m+3)*pow(T,3.0) +
                th.NASA7(n,m+4)*pow(T,4.0) )*th.Ru/th.MW(n);

    hi(n)  =(th.NASA7(n,m+0)                  +
             th.NASA7(n,m+1)*    T      / 2.0 +
             th.NASA7(n,m+2)*pow(T,2.0) / 3.0 +
             th.NASA7(n,m+3)*pow(T,3.0) / 4.0 +
             th.NASA7(n,m+4)*pow(T,4.0) / 5.0 +
             th.NASA7(n,m+5)/    T            )*T*th.Ru/th.MW(n);

    cp += cps  *Y(n);
    h  += hi(n)*Y(n);
  }

  // Compute mixuture enthalpy
  gamma = cp/(cp-Rmix);

  // Mixture speed of sound
  c = sqrt(gamma*Rmix*T);

  // Compute density
  rho = p/(Rmix*T);

  // Compute momentum
  rhou = rho*u;
  rhov = rho*v;
  rhow = rho*w;
  // Compuute TKE
  tke = 0.5*(pow(u,2.0) +
             pow(v,2.0) +
             pow(w,2.0))*
                 rho    ;

  // Compute internal, total, energy
  e = h - p/rho;
  rhoE = rho*e + tke;

  // Set values of new properties
  // Density
  b.Q(i,j,k,0) = rho;
  // Momentum
  b.Q(i,j,k,1) = rhou;
  b.Q(i,j,k,2) = rhov;
  b.Q(i,j,k,3) = rhow;
  // Total Energy
  b.Q(i,j,k,4) = rhoE;
  // Species mass
  for (int n=0; n<ns-1; n++)
  {
    b.Q(i,j,k,5+n) = Y(n)*rho;
  }
  // gamma,cp,h,c,e,hi
  b.qh(i,j,k,0) = gamma;
  b.qh(i,j,k,1) = cp;
  b.qh(i,j,k,2) = rho*h;
  b.qh(i,j,k,3) = c;
  b.qh(i,j,k,4) = rho*e;
  for (int n=0; n<=ns-1; n++)
  {
    b.qh(i,j,k,5+n) = hi(n);
  }

#ifndef NSCOMPILE
  token.release(id);
#endif
  });
  }


  else if ( given.compare("cons") == 0 )
  {
  Kokkos::parallel_for("Compute primatives from conserved quantities via tpg",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {
#ifndef NSCOMPILE
  int id = token.acquire();
#endif

  // Updates all primatives from conserved quantities
  // Along the way, we need to compute mixture properties
  // gamma, cp, h, e, hi
  // So we store these as well.

  double& rho = b.Q(i,j,k,0);
  double& rhou = b.Q(i,j,k,1);
  double& rhov = b.Q(i,j,k,2);
  double& rhow = b.Q(i,j,k,3);
  double& rhoE = b.Q(i,j,k,4);

  double p;
  double e,tke;
  double T;
#ifdef NSCOMPILE
  double Y[ns];
#endif
  double gamma,cp,h,c;
#ifdef NSCOMPILE
  double hi[ns];
#endif
  double Rmix;

  // Compute TKE
  tke = 0.5*(pow(rhou,2.0) +
             pow(rhov,2.0) +
             pow(rhow,2.0))/
                 rho       ;

  // Compute species mass fraction
  Y(ns-1) = 1.0;
  for (int n=0; n<ns-1; n++)
  {
    Y(n) = b.Q(i,j,k,5+n)/b.Q(i,j,k,0);
    Y(ns-1) -= Y(n);
  }
  Y(ns-1) = fmax(0.0,Y(ns-1));

  // Internal energy
  e = (rhoE - tke)/rho;

  // Iterate on to find temperature
  int nitr=0, maxitr = 100;
  double tol = 1e-8;
  double error = 1e100;

  // Compute Rmix
  Rmix = 0.0;
  for (int n=0; n<=ns-1; n++)
  {
    Rmix += Y(n)/th.MW(n);
  }
  Rmix *= th.Ru;

  // Newtons method to find T
  T = ( b.q(i,j,k,4) < 1.0 ) ? 300.0 : b.q(i,j,k,4); // Initial guess of T
  while( (abs(error) > tol) && (nitr < maxitr))
  {
    h = 0.0;
    cp = 0.0;
    for (int n=0; n<=ns-1; n++)
    {
      int m = ( T <= th.NASA7(n,0) ) ? 8 : 1;

      double cps=(th.NASA7(n,m+0)            +
                  th.NASA7(n,m+1)*    T      +
                  th.NASA7(n,m+2)*pow(T,2.0) +
                  th.NASA7(n,m+3)*pow(T,3.0) +
                  th.NASA7(n,m+4)*pow(T,4.0) )*th.Ru/th.MW(n);

      hi(n)  =(th.NASA7(n,m+0)                  +
               th.NASA7(n,m+1)*    T      / 2.0 +
               th.NASA7(n,m+2)*pow(T,2.0) / 3.0 +
               th.NASA7(n,m+3)*pow(T,3.0) / 4.0 +
               th.NASA7(n,m+4)*pow(T,4.0) / 5.0 +
               th.NASA7(n,m+5)/    T            )*T*th.Ru/th.MW(n);

      cp += cps  *Y(n);
      h  += hi(n)*Y(n);
    }

    error = e - (h - Rmix*T);
    T = T - error/(-cp - Rmix);
    nitr += 1;
  }

  // Compute mixuture pressure
  p = rho*Rmix*T;
  // Compute mixture gamma
  gamma = cp/(cp-Rmix);

  // Mixture speed of sound
  c = sqrt(gamma*Rmix*T);

  // Set values of new properties
  // Pressure, temperature, Y
  b.q(i,j,k,0) = p;
  b.q(i,j,k,1) = rhou/rho;
  b.q(i,j,k,2) = rhov/rho;
  b.q(i,j,k,3) = rhow/rho;
  b.q(i,j,k,4) = T;
  for (int n=0; n<ns-1; n++)
  {
    b.q(i,j,k,5+n) = Y(n);
  }
  // gamma,cp,h,c,e,hi
  b.qh(i,j,k,0) = gamma;
  b.qh(i,j,k,1) = cp;
  b.qh(i,j,k,2) = rho*h;
  b.qh(i,j,k,3) = c;
  b.qh(i,j,k,4) = rho*e;
  for (int n=0; n<=ns-1; n++)
  {
    b.qh(i,j,k,5+n) = hi(n);
  }

#ifndef NSCOMPILE
  token.release(id);
#endif
  });
  }
  else
  {
  throw std::invalid_argument( "Invalid given string in tpg.");
  }
}

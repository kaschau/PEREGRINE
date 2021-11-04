#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "thtrdat_.hpp"
#include "compute.hpp"
#include <math.h>
#include <stdexcept>

void tpg(block_ b,
   const thtrdat_ th,
   const int face,
   const std::string given,
   const int i/*=0*/,
   const int j/*=0*/,
   const int k/*=0*/) {

  MDRange3 range = get_range3(b, face, i, j, k);
  Kokkos::Experimental::UniqueToken<exec_space> token;
  const int numIds = token.size();

  const int ns=th.ns;
  twoDview Y("Y", ns, numIds);
  twoDview hi("hi", ns, numIds);
  twoDview cps("cps", ns, numIds);

  if ( given.compare("prims") == 0 )
  {
  twoDview rhoY("rhoY", ns, numIds);
  Kokkos::parallel_for("Compute all conserved quantities from primatives via tgp",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {
  int id = token.acquire();

  // Updates all conserved quantities from primatives
  // Along the way, we need to compute mixture properties
  // gamma, cp, h, e, hi
  // So we store these as well.

  double p;
  double u,v,w,tke;
  double T;

  double rho,rhoinv;
  double rhou,rhov,rhow;
  double e,rhoE;
  double gamma,cp,h,c;
  double Rmix;

  p = b.q(i,j,k,0);
  u = b.q(i,j,k,1);
  v = b.q(i,j,k,2);
  w = b.q(i,j,k,3);
  T = b.q(i,j,k,4);
  // Compute nth species Y
  Y(ns-1,id) = 1.0;
  for (int n=0; n<ns-1; n++)
  {
    Y(n,id) = b.q(i,j,k,5+n);
    Y(ns-1,id) -= Y(n,id);
  }
  Y(ns-1,id) = fmax(0.0,Y(ns-1,id));

  // Compute Rmix
  Rmix = 0.0;
  for (int n=0; n<=ns-1; n++)
  {
    Rmix += th.Ru  *Y(n,id)/th.MW[n];
  }

  // Update mixture properties
  h    = 0.0;
  cp   = 0.0;
  int m;
  for (int n=0; n<=ns-1; n++)
  {
    m = ( T <= th.NASA7[n][0] ) ? 8 : 1;

    cps(n,id) =(th.NASA7[n][m+0]            +
             th.NASA7[n][m+1]*    T      +
             th.NASA7[n][m+2]*pow(T,2.0) +
             th.NASA7[n][m+3]*pow(T,3.0) +
             th.NASA7[n][m+4]*pow(T,4.0) )*th.Ru/th.MW[n];

    hi(n,id)  =(th.NASA7[n][m+0]                  +
             th.NASA7[n][m+1]*    T      / 2.0 +
             th.NASA7[n][m+2]*pow(T,2.0) / 3.0 +
             th.NASA7[n][m+3]*pow(T,3.0) / 4.0 +
             th.NASA7[n][m+4]*pow(T,4.0) / 5.0 +
             th.NASA7[n][m+5]/    T            )*T*th.Ru/th.MW[n];

    cp   += cps(n,id)*Y(n,id);
    h    +=  hi(n,id)*Y(n,id);
  }

  // Compute mixuture enthalpy
  gamma = cp/(cp-Rmix);

  // Mixture speed of sound
  c = sqrt(gamma*Rmix*T);

  // Compute density
  rho = p/(Rmix*T);
  rhoinv = 1.0/rho;

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
  e = h - p*rhoinv;
  rhoE = rho*e + tke;

  // Compute species mass
  for (int n=0; n<=ns-1; n++)
  {
    rhoY(n,id) = Y(n,id)*rho;
  }

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
    b.Q(i,j,k,5+n) = rhoY(n,id);
  }
  // gamma,cp,h,c,e,hi
  b.qh(i,j,k,0) = gamma;
  b.qh(i,j,k,1) = cp;
  b.qh(i,j,k,2) = rho*h;
  b.qh(i,j,k,3) = c;
  b.qh(i,j,k,4) = rho*e;
  for (int n=0; n<=ns-1; n++)
  {
    b.qh(i,j,k,5+n) = hi(n,id);
  }

  token.release(id);
  });
  }
  else if ( given.compare("cons") == 0 )
  {
  twoDview rhoY("rhoY", ns, numIds);
  Kokkos::parallel_for("Compute primatives from conserved quantities via tpg",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {
  int id = token.acquire();

  // Updates all primatives from conserved quantities
  // Along the way, we need to compute mixture properties
  // gamma, cp, h, e, hi
  // So we store these as well.

  double rho,rhoinv;
  double rhou,rhov,rhow;
  double e,rhoE;

  double p;
  double tke;
  double T;
  double gamma,cp,h,c;
  double Rmix;

  rho = b.Q(i,j,k,0);
  rhoinv = 1.0/b.Q(i,j,k,0);
  rhou = b.Q(i,j,k,1);
  rhov = b.Q(i,j,k,2);
  rhow = b.Q(i,j,k,3);
  // Compute TKE
  tke = 0.5*(pow(rhou,2.0) +
             pow(rhov,2.0) +
             pow(rhow,2.0))*
                 rhoinv    ;
  rhoE = b.Q(i,j,k,4);

  // Compute species mass fraction
  Y(ns-1,id) = 1.0;
  for (int n=0; n<ns-1; n++)
  {
    Y(n,id) = b.Q(i,j,k,5+n)/b.Q(i,j,k,0);
    Y(ns-1,id) -= Y(n,id);
  }
  Y(ns-1,id) = fmax(0.0,Y(ns-1,id));

  // Internal energy
  e = (rhoE - tke)*rhoinv;

  // Iterate on to find temperature
  int nitr=0, maxitr = 100;
  double tol = 1e-8;
  double error = 1e100;

  // Compute Rmix
  Rmix = 0.0;
  for (int n=0; n<=ns-1; n++)
  {
    Rmix += th.Ru  *Y(n,id)/th.MW[n];
  }

  // Newtons method to find T
  T = ( b.q(i,j,k,4) < 1.0 ) ? 300.0 : b.q(i,j,k,4); // Initial guess of T
  while( (abs(error) > tol) && (nitr < maxitr))
  {
    h = 0.0;
    cp = 0.0;
    for (int n=0; n<=ns-1; n++)
    {
      int m = ( T <= th.NASA7[n][0] ) ? 8 : 1;

      cps(n,id) =(th.NASA7[n][m+0]            +
               th.NASA7[n][m+1]*    T      +
               th.NASA7[n][m+2]*pow(T,2.0) +
               th.NASA7[n][m+3]*pow(T,3.0) +
               th.NASA7[n][m+4]*pow(T,4.0) )*th.Ru/th.MW[n];

      hi(n,id)  =(th.NASA7[n][m+0]                  +
               th.NASA7[n][m+1]*    T      / 2.0 +
               th.NASA7[n][m+2]*pow(T,2.0) / 3.0 +
               th.NASA7[n][m+3]*pow(T,3.0) / 4.0 +
               th.NASA7[n][m+4]*pow(T,4.0) / 5.0 +
               th.NASA7[n][m+5]/    T            )*T*th.Ru/th.MW[n];

      cp   += cps(n,id)*Y(n,id);
      h    +=  hi(n,id)*Y(n,id);
    }

    T = T - (e - (h - Rmix*T))/(-cp - Rmix);
    error = e - (h - Rmix*T);
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
    b.q(i,j,k,5+n) = Y(n,id);
  }
  // gamma,cp,h,c,e,hi
  b.qh(i,j,k,0) = gamma;
  b.qh(i,j,k,1) = cp;
  b.qh(i,j,k,2) = rho*h;
  b.qh(i,j,k,3) = c;
  b.qh(i,j,k,4) = rho*e;
  for (int n=0; n<=ns-1; n++)
  {
    b.qh(i,j,k,5+n) = hi(n,id);
  }

  token.release(id);
  });
  }
  else
  {
  throw std::invalid_argument( "Invalid given string in tpg.");
  }
}

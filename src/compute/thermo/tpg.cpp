#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "thermdat_.hpp"
#include "compute.hpp"
#include <math.h>
#include <numeric>
#include <stdexcept>

void tpg(block_ b,
      thermdat_ th,
    std::string face,
    std::string given) {

  MDRange3 range = get_range3(b, face);


  if ( given.compare("prims") == 0 )
  {
  Kokkos::parallel_for("Compute all conserved quantities from primatives",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

  // Updates all conserved quantities from primatives
  // Along the way, we need to compute mixture properties
  // gamma, cp, h
  // So we store these as well.

  int ns=th.ns;
  double p;
  double u,v,w,tke;
  double T;
  double Y[ns];

  double rho,rhoinv;
  double rhou,rhov,rhow;
  double e,rhoE;
  double rhoY[ns];
  double gamma,cp=0.0,h=0.0;
  double Rmix=0.0;

  double cps[ns],hs[ns];

  p = b.q(i,j,k,0);
  u = b.q(i,j,k,1);
  v = b.q(i,j,k,2);
  w = b.q(i,j,k,3);
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
  for (int n=0; n<=ns-1; n++)
  {
    int m = ( T <= th.N7[n][0] ) ? 8 : 1;

    cps[n] = th.N7[n][m+0]            +
             th.N7[n][m+1]*    T      +
             th.N7[n][m+2]*pow(T,2.0) +
             th.N7[n][m+3]*pow(T,3.0) +
             th.N7[n][m+4]*pow(T,4.0) ;

    hs[n]  = th.N7[n][m+0]                  +
             th.N7[n][m+1]*    T      / 2.0 +
             th.N7[n][m+2]*pow(T,2.0) / 3.0 +
             th.N7[n][m+3]*pow(T,3.0) / 4.0 +
             th.N7[n][m+4]*pow(T,4.0) / 5.0 +
             th.N7[n][m+5]/    T            ;

    Rmix +=        th.Ru  *Y[n]/th.MW[n];
    cp   += cps[n]*th.Ru  *Y[n]/th.MW[n];
    h    +=  hs[n]*th.Ru*T*Y[n]/th.MW[n];
  }

  // Compute mixuture enthalpy
  gamma = cp/(cp-Rmix);

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
    rhoY[n] = Y[n]*rho;
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
    b.Q(i,j,k,5+n) = rhoY[n];
  }
  // gamma,cp,h
  b.qh(i,j,k,0) = gamma;
  b.qh(i,j,k,1) = cp;
  b.qh(i,j,k,2) = rho*h;

  });
  }
  else if ( given.compare("cons") == 0 )
  {
  Kokkos::parallel_for("Compute primatives from conserved quantities.",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

  // Updates all primatives from conserved quantities
  // Along the way, we need to compute mixture properties
  // gamma, cp, h
  // So we store these as well.

  int ns=th.ns;
  double rho,rhoinv;
  double rhou,rhov,rhow;
  double e,rhoE;
  double rhoY[ns];

  double p;
  double u,v,w,tke;
  double T;
  double Y[ns];
  double gamma,cp=0.0,h;
  double Rmix;

  double cps[ns],hs[ns];

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
  Y[ns-1] = 1.0;
  for (int n=0; n<ns-1; n++)
  {
    Y[n] = b.Q(i,j,k,5+n)/b.Q(i,j,k,0);
    Y[ns-1] -= Y[n];
  }
  Y[ns-1] = std::max(0.0,Y[ns-1]);

  // Internal energy
  e = (rhoE - tke)*rhoinv;

  // Iterate on to find temperature
  double Tmin=1, Tmax=10000;
  int nitr=0, maxitr = 100;
  double tol = 1e-8;
  double error = 1e100;

  // Dumb, slow bisection
  while( (abs(error) > tol) || (nitr >= maxitr) )
  {
    T = (Tmax + Tmin)/2.0;
    h = 0.0;
    Rmix = 0.0;
    for (int n=0; n<=ns-1; n++)
    {
      int m = ( T <= th.N7[n][0] ) ? 8 : 1;

      hs[n]  = th.N7[n][m+0]                  +
               th.N7[n][m+1]*    T      / 2.0 +
               th.N7[n][m+2]*pow(T,2.0) / 3.0 +
               th.N7[n][m+3]*pow(T,3.0) / 4.0 +
               th.N7[n][m+4]*pow(T,4.0) / 5.0 +
               th.N7[n][m+5]/    T            ;

      Rmix +=        th.Ru  *Y[n]/th.MW[n];
      h    +=  hs[n]*th.Ru*T*Y[n]/th.MW[n];
    }

    error = e - (h - Rmix*T);
    nitr += 1;

    if( error > 0.0 )
    {
      Tmin = T;
    }else
    {
      Tmax = T;
    }
  }

  // Compute mixuture properties
  for (int n=0; n<=ns-1; n++)
  {
    int m = ( T <= th.N7[n][0] ) ? 8 : 1;

    cps[n] = th.N7[n][m+0]            +
             th.N7[n][m+1]*    T      +
             th.N7[n][m+2]*pow(T,2.0) +
             th.N7[n][m+3]*pow(T,3.0) +
             th.N7[n][m+4]*pow(T,4.0) ;

    cp   += cps[n]*th.Ru  *Y[n]/th.MW[n];
  }

  // Compute mixuture pressure
  p = rho*Rmix*T;
  // Compute mixture gamma
  gamma = cp/(cp-Rmix);

  // Set values of new properties
  // Pressure, temperature, Y
  b.q(i,j,k,0) = p;
  b.q(i,j,k,1) = rhou/rho;
  b.q(i,j,k,2) = rhov/rho;
  b.q(i,j,k,3) = rhow/rho;
  b.q(i,j,k,4) = T;
  for (int n=0; n<ns-1; n++)
  {
    b.q(i,j,k,5+n) = Y[n];
  }
  // gamma,cp,h
  b.qh(i,j,k,0) = gamma;
  b.qh(i,j,k,1) = cp;
  b.qh(i,j,k,2) = rho*h;

  });
  }
  else
  {
  throw std::invalid_argument( "Invalid given string in cpg.");
  }
}

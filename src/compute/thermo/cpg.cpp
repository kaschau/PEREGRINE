#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "thtrdat_.hpp"
#include "compute.hpp"
#include <math.h>
#include <numeric>
#include <stdexcept>

void cpg(block_ b,
      thtrdat_ th,
           int  face,
   std::string  given) {

  MDRange3 range = get_range3(b, face);


  if ( given.compare("prims") == 0 )
  {
  Kokkos::parallel_for("Compute all conserved quantities from primatives via cpg",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

  // Updates all conserved quantities from primatives
  // Along the way, we need to compute mixture properties
  // gamma, cp, h, e, hi
  // So we store these as well.

  int ns=th.ns;
  double p;
  double u,v,w,tke;
  double T;
  double Y[ns];

  double rho,rhoinv;
  double rhou,rhov,rhow;
  double e,rhoE;
  double rhoY[ns],hi[ns];
  double gamma,cp,h,c;
  double Rmix;

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
  Rmix = 0.0;
  cp   = 0.0;
  for (int n=0; n<=ns-1; n++)
  {
    Rmix += Y[n]*th.Ru/th.MW[n];
    cp   += Y[n]*th.cp0[n];
  }
  // Compute mixuture enthalpy
  h = cp*T;
  gamma = cp/(cp-Rmix);

  // Mixture speed of soung
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
  // gamma,cp,h,c,e,hi
  b.qh(i,j,k,0) = gamma;
  b.qh(i,j,k,1) = cp;
  b.qh(i,j,k,2) = rho*h;
  b.qh(i,j,k,3) = c;
  b.qh(i,j,k,4) = rho*e;
  for (int n=0; n<=ns-1; n++)
  {
    b.qh(i,j,k,5+n) = T*th.cp0[n];
  }

  });
  }
  else if ( given.compare("cons") == 0 )
  {
  Kokkos::parallel_for("Compute primatives from conserved quantities via cpg",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

  // Updates all primatives from conserved quantities
  // Along the way, we need to compute mixture properties
  // gamma, cp, h, e, hi
  // So we store these as well.

  int ns=th.ns;
  double rho,rhoinv;
  double rhou,rhov,rhow;
  double e,rhoE;
  double rhoY[ns];

  double p;
  double u,v,w,tke;
  double T;
  double Y[ns],hi[ns];
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
  Y[ns-1] = 1.0;
  for (int n=0; n<ns-1; n++)
  {
    Y[n] = b.Q(i,j,k,5+n)/b.Q(i,j,k,0);
    Y[ns-1] -= Y[n];
  }
  Y[ns-1] = std::max(0.0,Y[ns-1]);

  // Internal energy
  e = (rhoE - tke)*rhoinv;

  // Compute mixuture cp
  Rmix = 0.0;
  cp   = 0.0;
  for (int n=0; n<=ns-1; n++)
  {
    Rmix += Y[n]*th.Ru/th.MW[n];
    cp   += Y[n]*th.cp0[n];
  }

  // Compute mixuture temperature,pressure
  T = e/(cp-Rmix);
  p = rho*Rmix*T;

  // Compute mixture enthalpy
  h = e + p*rhoinv;
  gamma = cp/(cp-Rmix);

  // Mixture speed of soung
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
    b.q(i,j,k,5+n) = Y[n];
  }
  // gamma,cp,h,c,e,hi
  b.qh(i,j,k,0) = gamma;
  b.qh(i,j,k,1) = cp;
  b.qh(i,j,k,2) = rho*h;
  b.qh(i,j,k,3) = c;
  b.qh(i,j,k,4) = rho*e;
  for (int n=0; n<=ns-1; n++)
  {
    b.qh(i,j,k,5+n) = T*th.cp0[n];
  }

  });
  }
  else
  {
  throw std::invalid_argument( "Invalid given string in cpg.");
  }
}

#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "thermdat_.hpp"
#include "compute.hpp"
#include <math.h>
#include <numeric>
#include <stdexcept>

void calEOS_perfect(block_ b,
                 thermdat_ th,
               std::string face,
               std::string given) {

  MDRange3 range = get_range3(b, face);


  if ( given.compare("PT") == 0 )
  {
  Kokkos::parallel_for("Compute total energy from temperature, momentum, density",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

  double ys[th.ns];
  double T,p,rhoinv;
  double rhou,rhov,rhow,tke;
  double cp=0.0,h,Rmix=0.0;
  int ns;

  T = b.q(i,j,k,4);
  p = b.q(i,j,k,0);
  rhoinv = 1.0/b.Q(i,j,k,0);
  rhou = b.Q(i,j,k,1);
  rhov = b.Q(i,j,k,1);
  rhow = b.Q(i,j,k,1);
  ns = th.ns;

  // Compute nth species Y
  ys[ns] = 1.0;
  for (int n=0; n<ns; n++)
  {
    ys[n] = b.Q(i,j,k,5+1+n)*rhoinv;
    ys[ns] -= ys[n];
  }
  ys[th.ns] = std::max(0.0,ys[ns]);

  // Compute mixuture cp
  for (int n=0; n<=ns; n++)
  {
    cp += ys[n]*th.cp0[n]/th.MW[n] * th.R;
    Rmix += th.R/th.MW[n];
  }
  // Compute mixuture enthalpy
  h = cp*T;

  // Compuute TKE
  tke = 0.5*(pow(rhou,2.0) +
             pow(rhov,2.0) +
             pow(rhow,2.0))*
                 rhoinv    ;

  // Set values of new properties
  // Total Energy
  b.Q(i,j,k,4) = h - p*rhoinv + tke;
  // gamma,cp,h
  b.qh(i,j,k,0) = cp/(cp-Rmix);
  b.qh(i,j,k,1) = cp;
  b.qh(i,j,k,2) = h;

  });
  }
  else if ( given.compare("Erho") == 0 )
  {
  Kokkos::parallel_for("Compute temperature from total energy, momentum, density",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {
  double ys[th.ns];
  double T,p,rho;
  double rhoE,e,rhoinv;
  double rhou,rhov,rhow,tke;
  double gamma,cp=0.0,h;
  double Rmix=0.0;
  int ns;

  rhoE = b.Q(i,j,k,4);
  rhoinv = 1.0/b.Q(i,j,k,0);
  rho = b.Q(i,j,k,0);
  rhou = b.Q(i,j,k,1);
  rhov = b.Q(i,j,k,1);
  rhow = b.Q(i,j,k,1);
  ns = th.ns;

  // Compuute TKE
  tke = 0.5*(pow(rhou,2.0) +
             pow(rhov,2.0) +
             pow(rhow,2.0))*
                 rhoinv    ;

  // Internal energy
  e = rhoE*rhoinv - tke;

  // Compute nth species Y
  ys[ns] = 1.0;
  for (int n=0; n<ns; n++)
  {
    ys[n] = b.Q(i,j,k,5+1+n)*rhoinv;
    ys[ns] -= ys[n];
  }
  ys[ns] = std::max(0.0,ys[ns]);

  // Compute mixuture cp
  for (int n=0; n<=ns; n++)
  {
    cp += ys[n]*th.cp0[n]/th.MW[n] * th.R;
    Rmix += th.R/th.MW[n];
  }
  // Compute mixuture temperature,pressure
  T = e/(cp-Rmix);
  p = rho*Rmix*T;

  // Compute mixture enthalpy
  h = e + p*rhoinv;
  gamma = cp/(cp-Rmix);



  // Set values of new properties
  // Pressure, temperature
  b.q(i,j,k,0) = p;
  b.q(i,j,k,4) = T;
  // gamma,cp,h
  b.qh(i,j,k,0) = gamma;
  b.qh(i,j,k,1) = cp;
  b.qh(i,j,k,2) = h;

  });
  }
  else
  {
  throw std::invalid_argument( "Invalid given string in calEOSperfect.");
  }
}

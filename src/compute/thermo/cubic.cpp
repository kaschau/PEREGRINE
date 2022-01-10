#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "thtrdat_.hpp"
#include "compute.hpp"
#include <math.h>
#include <stdexcept>

void cubic(block_ b,
   const thtrdat_ th,
   const int nface,
   const std::string given,
   const int indxI/*=0*/,
   const int indxJ/*=0*/,
   const int indxK/*=0*/) {

  MDRange3 range = get_range3(b, nface, indxI, indxJ, indxK);
  Kokkos::Experimental::UniqueToken<exec_space> token;
  const int numIds = token.size();

  const int ns=th.ns;
  twoDview Y("Y", ns, numIds);
  twoDview X("X", ns, numIds);
  twoDview X("ai", ns, numIds);
  twoDview hi("hi", ns, numIds);

  if ( given.compare("prims") == 0 )
  {
  twoDview rhoY("rhoY", ns, numIds);
  Kokkos::parallel_for("Compute all conserved quantities from primatives via real gas",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {
  int id = token.acquire();

  // Updates all conserved quantities from primatives
  // Along the way, we need to compute mixture properties
  // gamma, cp, h, e, hi
  // So we store these as well.

  double& p = b.q(i,j,k,0);
  double& u = b.q(i,j,k,1);
  double& v = b.q(i,j,k,2);
  double& w = b.q(i,j,k,3);
  double& T = b.q(i,j,k,4);

  double rho,rhoinv;
  double rhou,rhov,rhow;
  double e,tke,rhoE;
  double gamma,cp,cps,h,c;
  double Rmix;

  // Compute nth species Y
  Y(ns-1,id) = 1.0;
  for (int n=0; n<ns-1; n++)
  {
    Y(n,id) = b.q(i,j,k,5+n);
    Y(ns-1,id) -= Y(n,id);
  }
  Y(ns-1,id) = fmax(0.0,Y(ns-1,id));

  // Compute Rmix, and y->x denom
  Rmix = 0.0;
  denom= 0.0;
  for (int n=0; n<=ns-1; n++)
  {
    Rmix += th.Ru * Y(n,id)/th.MW(n);
    denom += Y(n,id)/th.MW(n);
  }

  // Compute mole fraction
  for (int n=0; n<=ns-1; n++)
  {
    X(n,id) = (Y(n,id)/th.MW(n))/denom;
  }

  // Real gas coefficients for cubic EOS
  // PRS
  constexpr double uRG=2.0, wRG=-1.0, bCoeff=0.09725, aCoeff=0.006679375, fw0=0.37464, fw1=1.54226, fw2=-0.26992;
  double bi,ai,fOmega,Tr;
  double am = 0.0, bm = 0.0;
  double Astar, Bstar;
  double z0,z1,z2, Z;

  // Compressibility Factor, Z
  for (int n=0; n<=ns-1; n++)
  {
    // PRS
    Tr = T/th.Tcrit(n);
    bi = bCoeff*th.Ru*th.Tcrit(n)/Pcrit(n);
    fOmega = fw0 + fw1*th.acentric(n) + fw2*pow(th.acentric,2.0);
    ai(n,id) = aCoeff*pow(th.Ru,2.0)*pow(th.Tcrit,2.5)/(Pc*sqrt(T)) * pow(1.0+fOmega*(1-pow(Tr,0.5)),2.0);

    bm += X(n,id)*b
  }
  for (int n=0; n<=ns-1; n++)
  {
    for (int n2=0; n2<=ns-1; n2++)
    {
      am += X(n,id)*X(n2,id)*sqrt(ai(n)*ai(n2)); // - (1 - kij)  <- For now we ignore binary interaciton coeff, i.e. assume kij=1
    }
  }

  Astar = am*p/pow(th.Ru*T,2.0);
  Bstar = bm*p/(th.Ru*T);

  // Solve cubic EOS for Z
  // https://www.e-education.psu.edu/png520/m11_p6.html
  double Bstar2 = pow(Bstar,2.0);
  z0 = - (Astar*Bstar + wRG*Bstar2 + wRG*Bstar2*Bstar);
  z1 = Astar + wRG*Bstar2 - uRG*Bstar - uRG*Bstar2;
  z2 = -(1.0 + Bstar - uRG*Bstar);

  double M,Q,R;

  Q = (pow(z2,2.0)-3.0*z1)/9.0;
  R = (2.0*pow(z3,3.0)-9.0*z2*z1+27.0*z0)/54.0;
  M = pow(R,2.0) - pow(Q,3.0);

  double z2o3 = z2/3.0:
  if (M > 0.0) {
    double S = -R/abs(R) * pow(abs(R)+sqrt(M),(1.0/3.0));
    Z = S + Q/S - z2o3;
  }else{
    double q1p5 = sqrt(pow(Q,3.0));
    double sqQ = sqrt(Q);
    double theta = acos(R/q1p5);
    double x1 = -(2.0*sqQ*cos(theta/3.0)-z2o3);
    double x2 = -(2.0*sqQ*cos((theta+2*3.14159265358979323846)/3.0)-z2o3);
    double x3 = -(2.0*sqQ*cos((theta-2*3.14159265358979323846)/3.0)-z2o3);

    Z = fmax({x1,x2,x3});
  }



  // Update mixture properties
  h    = 0.0;
  cp   = 0.0;
  int m;
  for (int n=0; n<=ns-1; n++)
  {
    m = ( T <= th.NASA7(n,0) ) ? 8 : 1;

    cps       =(th.NASA7(n,m+0)            +
                th.NASA7(n,m+1)*    T      +
                th.NASA7(n,m+2)*pow(T,2.0) +
                th.NASA7(n,m+3)*pow(T,3.0) +
                th.NASA7(n,m+4)*pow(T,4.0) )*th.Ru/th.MW(n);

    hi(n,id)  =(th.NASA7(n,m+0)                  +
                th.NASA7(n,m+1)*    T      / 2.0 +
                th.NASA7(n,m+2)*pow(T,2.0) / 3.0 +
                th.NASA7(n,m+3)*pow(T,3.0) / 4.0 +
                th.NASA7(n,m+4)*pow(T,4.0) / 5.0 +
                th.NASA7(n,m+5)/    T            )*T*th.Ru/th.MW(n);

    cp += cps      *Y(n,id);
    h  +=  hi(n,id)*Y(n,id);
  }

  // Compute mixuture enthalpy
  gamma = cp/(cp-Rmix);

  // Mixture speed of sound
  c = sqrt(gamma*Rmix*T);

  // Compute density
  rho = p/(Z*Rmix*T);
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
  Kokkos::parallel_for("Compute primatives from conserved quantities via real gas",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {
  int id = token.acquire();

  // Updates all primatives from conserved quantities
  // Along the way, we need to compute mixture properties
  // gamma, cp, h, e, hi
  // So we store these as well.

  double& rho = b.Q(i,j,k,0);
  double rhoinv = 1.0/rho;
  double& rhou = b.Q(i,j,k,1);
  double& rhov = b.Q(i,j,k,2);
  double& rhow = b.Q(i,j,k,3);
  double& rhoE = b.Q(i,j,k,4);

  double p;
  double e,tke;
  double T;
  double gamma,cp,cps,h,c;
  double Rmix;

  // Compute TKE
  tke = 0.5*(pow(rhou,2.0) +
             pow(rhov,2.0) +
             pow(rhow,2.0))*
                 rhoinv    ;

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
    Rmix += th.Ru  *Y(n,id)/th.MW(n);
  }

  // Newtons method to find T
  T = ( b.q(i,j,k,4) < 1.0 ) ? 300.0 : b.q(i,j,k,4); // Initial guess of T
  while( (abs(error) > tol) && (nitr < maxitr))
  {
    h = 0.0;
    cp = 0.0;
    for (int n=0; n<=ns-1; n++)
    {
      int m = ( T <= th.NASA7(n,0) ) ? 8 : 1;

      cps       =(th.NASA7(n,m+0)            +
                  th.NASA7(n,m+1)*    T      +
                  th.NASA7(n,m+2)*pow(T,2.0) +
                  th.NASA7(n,m+3)*pow(T,3.0) +
                  th.NASA7(n,m+4)*pow(T,4.0) )*th.Ru/th.MW(n);

      hi(n,id)  =(th.NASA7(n,m+0)                  +
                  th.NASA7(n,m+1)*    T      / 2.0 +
                  th.NASA7(n,m+2)*pow(T,2.0) / 3.0 +
                  th.NASA7(n,m+3)*pow(T,3.0) / 4.0 +
                  th.NASA7(n,m+4)*pow(T,4.0) / 5.0 +
                  th.NASA7(n,m+5)/    T            )*T*th.Ru/th.MW(n);

      cp += cps      *Y(n,id);
      h  +=  hi(n,id)*Y(n,id);
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
  throw std::invalid_argument( "Invalid given string in real gas.");
  }
}

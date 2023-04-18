#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "compute.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"
#include <math.h>
#include <stdexcept>
#include <string.h>

void cpg(block_ &b, const thtrdat_ &th, const int &nface,
         const std::string &given, const int &indxI /*=0*/,
         const int &indxJ /*=0*/, const int &indxK /*=0*/) {

// For performance purposes, we want to compile with ns known whenever possible
// however, for testing, developement, etc. we want the flexibility to
// have it at run time as well. So we define some macros here to allow that.
#ifndef NSCOMPILE
  Kokkos::Experimental::UniqueToken<execSpace> token;
  int numIds = token.size();
  const int ns = th.ns;
  twoDview Y("Y", numIds, ns);
#endif

#ifdef NSCOMPILE
#define Y(INDEX) Y[INDEX]
#define ns NS
#else
#define Y(INDEX) Y(id, INDEX)
#endif

  MDRange3 range = getRange3(b, nface, indxI, indxJ, indxK);
  if (given.compare("prims") == 0) {
    Kokkos::parallel_for(
        "Compute all conserved quantities from primatives via cpg", range,
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
#ifndef NSCOMPILE
          int id = token.acquire();
#endif

          // Updates all conserved quantities from primatives
          // Along the way, we need to compute mixture properties
          // gamma, cp, h, e
          // So we store these as well.

          const double &p = b.q(i, j, k, 0);
          const double &u = b.q(i, j, k, 1);
          const double &v = b.q(i, j, k, 2);
          const double &w = b.q(i, j, k, 3);
          const double &T = b.q(i, j, k, 4);
#ifdef NSCOMPILE
          double Y(ns);
#endif

          double rho;
          double rhou, rhov, rhow;
          double e, tke, rhoE;
          double gamma, cp, h, c;
          double Rmix;

          // Compute nth species Y
          Y(ns - 1) = 1.0;
          double testSum = 0.0;
          for (int n = 0; n < ns - 1; n++) {
            b.q(i, j, k, 5 + n) = fmax(fmin(b.q(i, j, k, 5 + n), 1.0), 0.0);
            Y(n) = b.q(i, j, k, 5 + n);
            Y(ns - 1) -= Y(n);
            testSum += Y(n);
          }

          // Renormalize if necessary
          if (testSum > 1.0) {
            Y(ns - 1) = 0.0;
            for (int n = 0; n < ns - 1; n++) {
              Y(n) /= testSum;
            }
          }

          // Update mixture properties
          Rmix = 0.0;
          cp = 0.0;
          for (int n = 0; n <= ns - 1; n++) {
            Rmix += Y(n) / th.MW(n);
            cp += Y(n) * th.cp0(n);
          }
          Rmix *= th.Ru;

          // Compute mixuture enthalpy
          h = cp * T;
          gamma = cp / (cp - Rmix);

          // Mixture speed of soung
          c = sqrt(gamma * Rmix * T);

          // Compute density
          rho = p / (Rmix * T);

          // Compute momentum
          rhou = rho * u;
          rhov = rho * v;
          rhow = rho * w;
          // Compuute TKE
          tke = 0.5 * (pow(u, 2.0) + pow(v, 2.0) + pow(w, 2.0)) * rho;

          // Compute internal, total, energy
          e = h - p / rho;
          rhoE = rho * e + tke;

          // Set values of new properties
          // Density
          b.Q(i, j, k, 0) = rho;
          // Momentum
          b.Q(i, j, k, 1) = rhou;
          b.Q(i, j, k, 2) = rhov;
          b.Q(i, j, k, 3) = rhow;
          // Total Energy
          b.Q(i, j, k, 4) = rhoE;
          // Species mass
          for (int n = 0; n < ns - 1; n++) {
            b.Q(i, j, k, 5 + n) = Y(n) * rho;
          }
          // gamma,cp,h,c,e,hi
          b.qh(i, j, k, 0) = gamma;
          b.qh(i, j, k, 1) = cp;
          b.qh(i, j, k, 2) = rho * h;
          b.qh(i, j, k, 3) = c;
          b.qh(i, j, k, 4) = rho * e;
          for (int n = 0; n <= ns - 1; n++) {
            b.qh(i, j, k, 5 + n) = T * th.cp0(n);
          }

#ifndef NSCOMPILE
          token.release(id);
#endif
        });
  } else if (given.compare("cons") == 0) {
    Kokkos::parallel_for(
        "Compute primatives from conserved quantities via cpg", range,
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
#ifndef NSCOMPILE
          int id = token.acquire();
#endif

          // Updates all primatives from conserved quantities
          // Along the way, we need to compute mixture properties
          // gamma, cp, h, e, hi
          // So we store these as well.

          const double &rho = b.Q(i, j, k, 0);
          const double &rhou = b.Q(i, j, k, 1);
          const double &rhov = b.Q(i, j, k, 2);
          const double &rhow = b.Q(i, j, k, 3);
          const double &rhoE = b.Q(i, j, k, 4);

          double p;
          double T;
          double e, tke;
#ifdef NSCOMPILE
          double Y[ns];
#endif
          double gamma, cp, h, c;
          double Rmix;

          // Compute TKE
          tke = 0.5 * (pow(rhou, 2.0) + pow(rhov, 2.0) + pow(rhow, 2.0)) / rho;

          // Compute species mass fraction
          Y(ns - 1) = 1.0;
          double testSum = 0.0;
          for (int n = 0; n < ns - 1; n++) {
            b.Q(i, j, k, 5 + n) =
                fmax(fmin(b.Q(i, j, k, 5 + n), b.Q(i, j, k, 0)), 0.0);
            Y(n) = b.Q(i, j, k, 5 + n) / b.Q(i, j, k, 0);
            Y(ns - 1) -= Y(n);
            testSum += Y(n);
          }

          // Renormalize if necessary
          if (testSum > 1.0) {
            Y(ns - 1) = 0.0;
            for (int n = 0; n < ns - 1; n++) {
              Y(n) /= testSum;
              b.Q(i, j, k, 5 + n) = Y(n) * b.Q(i, j, k, 0);
            }
          }

          // Internal energy
          e = (rhoE - tke) / rho;

          // Compute mixuture cp
          Rmix = 0.0;
          cp = 0.0;
          for (int n = 0; n <= ns - 1; n++) {
            Rmix += Y(n) / th.MW(n);
            cp += Y(n) * th.cp0(n);
          }
          Rmix *= th.Ru;

          // Compute mixuture temperature,pressure
          T = e / (cp - Rmix);
          p = rho * Rmix * T;

          // Compute mixture enthalpy
          h = e + p / rho;
          gamma = cp / (cp - Rmix);

          // Mixture speed of soung
          c = sqrt(gamma * Rmix * T);

          // Set values of new properties
          // Pressure, temperature, Y
          b.q(i, j, k, 0) = p;
          b.q(i, j, k, 1) = rhou / rho;
          b.q(i, j, k, 2) = rhov / rho;
          b.q(i, j, k, 3) = rhow / rho;
          b.q(i, j, k, 4) = T;
          for (int n = 0; n < ns - 1; n++) {
            b.q(i, j, k, 5 + n) = Y(n);
          }
          // gamma,cp,h,c,e,hi
          b.qh(i, j, k, 0) = gamma;
          b.qh(i, j, k, 1) = cp;
          b.qh(i, j, k, 2) = rho * h;
          b.qh(i, j, k, 3) = c;
          b.qh(i, j, k, 4) = rho * e;
          for (int n = 0; n <= ns - 1; n++) {
            b.qh(i, j, k, 5 + n) = T * th.cp0(n);
          }

#ifndef NSCOMPILE
          token.release(id);
#endif
        });
  } else {
    throw std::invalid_argument("Invalid given string in cpg.");
  }
}

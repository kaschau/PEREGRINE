#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>
#include <math.h>

PG_ABI void pgCpgFromPrims(int count, pgOut *Q_, pgOut *q_, pgOut *qh_,
                           const pgIn &MW_, const pgIn &cp0_, double Ru,
                           const pgRange *r) {
  for (int e = 0; e < count; e++) {
    auto Q = as4(Q_[e]);
    auto q = as4(q_[e]);
    auto qh = as4(qh_[e]);
    auto MW = as1(MW_);
    auto cp0 = as1(cp0_);

    MDRange3 range = range3(r[e]);
    Kokkos::parallel_for(
        "Compute all conserved quantities from primatives via cpg", range,
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
          // Updates all conserved quantities from primatives
          // Along the way, we need to compute mixture properties
          // gamma, cp, h, e
          // So we store these as well.

          const double &p = q(i, j, k, 0);
          const double &u = q(i, j, k, 1);
          const double &v = q(i, j, k, 2);
          const double &w = q(i, j, k, 3);
          const double &T = q(i, j, k, 4);
          double Y[ns];

          double rho;
          double rhou, rhov, rhow;
          double e, tke, rhoE;
          double gamma, cp, h, c;
          double Rmix;

          // Compute nth species Y
          Y[ns - 1] = 1.0;
          double testSum = 0.0;
          for (int n = 0; n < ns - 1; n++) {
            q(i, j, k, 5 + n) = fmax(fmin(q(i, j, k, 5 + n), 1.0), 0.0);
            Y[n] = q(i, j, k, 5 + n);
            Y[ns - 1] -= Y[n];
            testSum += Y[n];
          }

          // Renormalize if necessary
          if (testSum > 1.0) {
            Y[ns - 1] = 0.0;
            for (int n = 0; n < ns - 1; n++) {
              Y[n] /= testSum;
            }
          }

          // Update mixture properties
          Rmix = 0.0;
          cp = 0.0;
          for (int n = 0; n <= ns - 1; n++) {
            Rmix += Y[n] / MW(n);
            cp += Y[n] * cp0(n);
          }
          Rmix *= Ru;

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
          Q(i, j, k, 0) = rho;
          // Momentum
          Q(i, j, k, 1) = rhou;
          Q(i, j, k, 2) = rhov;
          Q(i, j, k, 3) = rhow;
          // Total Energy
          Q(i, j, k, 4) = rhoE;
          // Species mass
          for (int n = 0; n < ns - 1; n++) {
            Q(i, j, k, 5 + n) = Y[n] * rho;
          }
          // gamma,cp,h,c,e,hi
          qh(i, j, k, 0) = gamma;
          qh(i, j, k, 1) = cp;
          qh(i, j, k, 2) = rho * h;
          qh(i, j, k, 3) = c;
          qh(i, j, k, 4) = rho * e;
          for (int n = 0; n <= ns - 1; n++) {
            qh(i, j, k, 5 + n) = T * cp0(n);
          }
        });
  }
}

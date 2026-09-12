#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>
#include <math.h>

PG_ABI void pgTpgFromPrims(int count, const pgView *Q_, const pgView *q_,
                           const pgView *qh_, const pgView &MW_,
                           const pgView &cpPoly_, const pgView &hPoly_,
                           const pgView &hRef_, double Ru, const pgRange *r) {
  for (int e = 0; e < count; e++) {
    auto Q = as4(Q_[e]), q = as4(q_[e]), qh = as4(qh_[e]);
    auto MW = as1(MW_);
    auto cpPoly = as2(cpPoly_);
    auto hPoly = as2(hPoly_);
    auto hRef = as1(hRef_);

    MDRange3 range = range3(r[e]);
    Kokkos::parallel_for(
        "Compute all conserved quantities from primatives via tgp", range,
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
          // Updates all conserved quantities from primatives
          // Along the way, we need to compute mixture properties
          // gamma, cp, h, e, hi
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
          double hi[ns];
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

          // Compute Rmix
          Rmix = 0.0;
          for (int n = 0; n <= ns - 1; n++) {
            Rmix += Y[n] / MW(n);
          }
          Rmix *= Ru;

          // Update mixture properties
          h = 0.0;
          cp = 0.0;
          {
            const double u = log(T);
            for (int n = 0; n <= ns - 1; n++) {
              // cp/R and h/(RT), Horner in u
              double cpR = 0.0, hRT = 0.0;
              for (int m = cpPoly.extent(1) - 1; m >= 0; m--)
                cpR = cpR * u + cpPoly(n, m);
              for (int m = hPoly.extent(1) - 1; m >= 0; m--)
                hRT = hRT * u + hPoly(n, m);
              hRT += hRef(n) / T;
              const double Rn = Ru / MW(n);
              hi[n] = hRT * T * Rn;
              cp += cpR * Rn * Y[n];
              h += hi[n] * Y[n];
            }
          }

          // Compute mixuture enthalpy
          gamma = cp / (cp - Rmix);

          // Mixture speed of sound
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
            qh(i, j, k, 5 + n) = hi[n];
          }
        });
  }
}

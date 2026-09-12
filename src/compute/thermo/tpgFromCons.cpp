#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>
#include <math.h>

PG_ABI void pgTpgFromCons(int count, const pgView *Q_, const pgView *q_,
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
        "Compute primatives from conserved quantities via tpg", range,
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
          // Updates all primatives from conserved quantities
          // Along the way, we need to compute mixture properties
          // gamma, cp, h, e, hi
          // So we store these as well.

          const double &rho = Q(i, j, k, 0);
          const double &rhou = Q(i, j, k, 1);
          const double &rhov = Q(i, j, k, 2);
          const double &rhow = Q(i, j, k, 3);
          const double &rhoE = Q(i, j, k, 4);

          double p;
          double e, tke;
          double T;
          double Y[ns];
          double gamma, cp, h, c;
          double hi[ns];
          double Rmix;

          // Compute TKE
          tke = 0.5 * (pow(rhou, 2.0) + pow(rhov, 2.0) + pow(rhow, 2.0)) / rho;

          // Compute species mass fraction
          Y[ns - 1] = 1.0;
          double testSum = 0.0;
          for (int n = 0; n < ns - 1; n++) {
            Q(i, j, k, 5 + n) =
                fmax(fmin(Q(i, j, k, 5 + n), Q(i, j, k, 0)), 0.0);
            Y[n] = Q(i, j, k, 5 + n) / Q(i, j, k, 0);
            Y[ns - 1] -= Y[n];
            testSum += Y[n];
          }

          // Renormalize if necessary
          if (testSum > 1.0) {
            Y[ns - 1] = 0.0;
            for (int n = 0; n < ns - 1; n++) {
              Y[n] /= testSum;
              Q(i, j, k, 5 + n) = Y[n] * Q(i, j, k, 0);
            }
          }

          // Internal energy
          e = (rhoE - tke) / rho;

          // Compute Rmix
          Rmix = 0.0;
          for (int n = 0; n <= ns - 1; n++) {
            Rmix += Y[n] / MW(n);
          }
          Rmix *= Ru;

          // Iterate on to find temperature
          int nitr = 0, maxitr = 100;
          double tol = 1e-8;
          double error = 1e100;
          // Newtons method to find T
          T = (q(i, j, k, 4) < 1.0) ? 300.0
                                    : q(i, j, k, 4); // Initial guess of T
          while ((abs(error) > tol) && (nitr < maxitr)) {
            h = 0.0;
            cp = 0.0;
            {
              const double u = log(T);
              for (int n = 0; n <= ns - 1; n++) {
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

            error = e - (h - Rmix * T);
            T = T - error / (-cp + Rmix);
            nitr += 1;
          }

          // Compute mixuture pressure
          p = rho * Rmix * T;
          // Compute mixture gamma
          gamma = cp / (cp - Rmix);

          // Mixture speed of sound
          c = sqrt(gamma * Rmix * T);

          // Set values of new properties
          // Pressure, temperature, Y
          q(i, j, k, 0) = p;
          q(i, j, k, 1) = rhou / rho;
          q(i, j, k, 2) = rhov / rho;
          q(i, j, k, 3) = rhow / rho;
          q(i, j, k, 4) = T;
          for (int n = 0; n < ns - 1; n++) {
            q(i, j, k, 5 + n) = Y[n];
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

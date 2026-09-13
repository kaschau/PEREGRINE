#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>
#include <math.h>

PG_ABI void pgCpgFromCons(int count, pgOut *Q_, pgOut *q_, pgOut *qh_,
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
        "Compute primatives from conserved quantities via cpg", range,
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
          double T;
          double e, tke;
          double Y[ns];
          double gamma, cp, h, c;
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

          // Compute mixuture cp
          Rmix = 0.0;
          cp = 0.0;
          for (int n = 0; n <= ns - 1; n++) {
            Rmix += Y[n] / MW(n);
            cp += Y[n] * cp0(n);
          }
          Rmix *= Ru;

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
            qh(i, j, k, 5 + n) = T * cp0(n);
          }
        });
  }
}

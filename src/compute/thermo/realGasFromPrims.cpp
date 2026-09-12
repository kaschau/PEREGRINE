#include "cubic.hpp"
#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>
#include <math.h>

PG_ABI void pgRealGasFromPrims(int count, const pgView *Q_, const pgView *q_,
                               const pgView *qh_, const pgView &MW_,
                               const pgView &cpPoly_, const pgView &hPoly_,
                               const pgView &hRef_, const pgView &Tcrit_,
                               const pgView &pcrit_, const pgView &acentric_,
                               double Ru, const pgRange *r) {
  for (int e = 0; e < count; e++) {
    auto Q = as4(Q_[e]), q = as4(q_[e]), qh = as4(qh_[e]);
    auto MW = as1(MW_);
    auto cpPoly = as2(cpPoly_);
    auto hPoly = as2(hPoly_);
    auto hRef = as1(hRef_);
    auto Tcrit = as1(Tcrit_);
    auto pcrit = as1(pcrit_);
    auto acentric = as1(acentric_);

    MDRange3 range = range3(r[e]);
    Kokkos::parallel_for(
        "Compute all conserved quantities from primatives via real gas", range,
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
          // Updates all conserved quantities from primatives
          // Along the way, we need to compute mixture properties
          // gamma, cp, h, e, hi
          // So we store these as well.

          const double p = q(i, j, k, 0);
          const double u = q(i, j, k, 1);
          const double v = q(i, j, k, 2);
          const double w = q(i, j, k, 3);
          const double T = q(i, j, k, 4);
          double Y[ns];
          double X[ns];

          double rho;
          double rhou, rhov, rhow;
          double e, tke, rhoE;
          double gamma, cp, h, c;
          double hi[ns];
          double Rmix;

          // Compute nth species Y
          Y[ns - 1] = 1.0;
          double testSum = 0.0;
          // Compute y->x denom
          double denom = 0.0;
          for (int n = 0; n < ns - 1; n++) {
            q(i, j, k, 5 + n) = fmax(fmin(q(i, j, k, 5 + n), 1.0), 0.0);
            Y[n] = q(i, j, k, 5 + n);
            Y[ns - 1] -= Y[n];
            denom += Y[n] / MW(n);
            testSum += Y[n];
          }
          denom += Y[ns - 1] / MW(ns - 1);

          // Renormalize if necessary
          if (testSum > 1.0) {
            Y[ns - 1] = 0.0;
            for (int n = 0; n < ns - 1; n++) {
              Y[n] /= testSum;
            }
          }

          // Compute mole fraction, mean molecular weight
          double MWmix = 0.0;
          for (int n = 0; n <= ns - 1; n++) {
            X[n] = (Y[n] / MW(n)) / denom;
            MWmix += MW(n) * X[n];
          }
          // Compute Rmix
          Rmix = Ru / MWmix;

          // Real gas coefficients for cubic EOS
          // -------------------------------------------------------------------------------------------------------------//
          // Peng-Robinson
          const double uRG = 2.0, wRG = -1.0, biConst = 0.077796,
                       aiConst = 0.457240, fw0 = 0.37464, fw1 = 1.54226,
                       fw2 = -0.26992;
          // -------------------------------------------------------------------------------------------------------------//

          // -------------------------------------------------------------------------------------------------------------//
          // Soave-Redlich-Kwong
          // const double uRG=1.0, wRG= 0.0, biConst=0.0866403,
          // aiConst=0.4274802, fw0=0.480  , fw1=1.574  , fw2=-0.176  ;
          // -------------------------------------------------------------------------------------------------------------//

          double am = 0.0, bm = 0.0;
          double ai[ns];
          double Astar, Bstar;

          // Compressibility Factor, Z
          for (int n = 0; n <= ns - 1; n++) {
            double Tr = T / Tcrit(n);
            double fOmega =
                fw0 + fw1 * acentric(n) + fw2 * pow(acentric(n), 2.0);
            double alpha = pow(1.0 + fOmega * (1 - sqrt(Tr)), 2.0);
            ai[n] = aiConst * (pow(Ru * Tcrit(n), 2.0) * alpha) / pcrit(n);
            double bi = biConst * (Ru * Tcrit(n)) / pcrit(n);

            bm += X[n] * bi;
          }
          for (int n = 0; n <= ns - 1; n++) {
            for (int n2 = 0; n2 <= ns - 1; n2++) {
              am += X[n] * X[n2] *
                    sqrt(ai[n] *
                         ai[n2]); // - (1 - kij)  <- For now we ignore binary
                                  // interaciton coeff, i.e. assume kij=1
            }
          }

          Astar = am * p / pow(Ru * T, 2.0);
          Bstar = bm * p / (Ru * T);

          // Solve cubic EOS for Z
          // https://www.e-education.psu.edu/png520/m11_p6.html
          double z0, z1, z2, Z;
          double Bstar2 = pow(Bstar, 2.0);
          z0 = -(Astar * Bstar + wRG * Bstar2 + wRG * Bstar2 * Bstar);
          z1 = Astar + wRG * Bstar2 - uRG * Bstar - uRG * Bstar2;
          z2 = -(1.0 + Bstar - uRG * Bstar);

          double cardanoQ, RR, M;
          cardanoQ = (pow(z2, 2.0) - 3.0 * z1) / 9.0;
          RR = (2.0 * pow(z2, 3.0) - 9.0 * z2 * z1 + 27.0 * z0) / 54.0;
          M = pow(RR, 2.0) - pow(cardanoQ, 3.0);

          double z2o3 = z2 / 3.0;
          if (M > 0.0) {
            double S = -RR / abs(RR) * pow(abs(RR) + sqrt(M), (1.0 / 3.0));
            Z = S + cardanoQ / S - z2o3;
          } else {
            double q1p5 = pow(cardanoQ, 1.5);
            double sqQ = sqrt(cardanoQ);
            double theta = acos(RR / q1p5);
            double x1 = -(2.0 * sqQ * cos(theta / 3.0)) - z2o3;
            double x2 =
                -(2.0 * sqQ * cos((theta + 2 * 3.14159265358979323846) / 3.0)) -
                z2o3;
            double x3 =
                -(2.0 * sqQ * cos((theta - 2 * 3.14159265358979323846) / 3.0)) -
                z2o3;

            Z = stableRoot(x1, x2, x3, Astar, Bstar, uRG, wRG);
          }

          // Update mixture properties

          // departure functions
          double dam = 0.0;
          for (int n = 0; n <= ns - 1; n++) {
            for (int n2 = 0; n2 <= ns - 1; n2++) {
              double fOmegaN =
                  fw0 + fw1 * acentric(n) + fw2 * pow(acentric(n), 2.0);
              double fOmegaN2 =
                  fw0 + fw1 * acentric(n2) + fw2 * pow(acentric(n2), 2.0);
              dam += X[n2] * X[n] * 1.0 *
                     (fOmegaN2 * sqrt(ai[n] * Tcrit(n2) / pcrit(n2)) +
                      fOmegaN * sqrt(ai[n2] * Tcrit(n) / pcrit(n)));
            }
          }
          dam *= -0.5 * Ru * sqrt(aiConst / T);
          double dAstardT = -2.0 * (Astar / T) * (1.0 - 0.5 * (T / am) * dam);
          double dBstardT = -Bstar / T;

          double dz0 = -(Bstar * dAstardT +
                         (Astar + (2.0 * Bstar + 3.0 * pow(Bstar, 2.0)) * wRG) *
                             dBstardT);
          double dz1 =
              (dAstardT + (2.0 * Bstar * (wRG - uRG) - uRG) * dBstardT);
          double dz2 = -(1.0 - uRG) * dBstardT;

          double dZdT = -(dz2 * pow(Z, 2.0) + dz1 * Z + dz0) /
                        (3.0 * pow(Z, 2.0) + 2.0 * Z * z2 + z1);

          double Cuw = 1.0 / (bm * Ru * sqrt(pow(uRG, 2.0) - 4.0 * wRG));
          double ZoB = Z / Bstar;
          double logZoB =
              log((2.0 * ZoB + (uRG - sqrt(pow(uRG, 2.0) - 4.0 * wRG))) /
                  (2.0 * ZoB + (uRG + sqrt(pow(uRG, 2.0) - 4.0 * wRG))));

          double cpDep = Cuw * (am / T - dam) * logZoB * 0.5 * T / am * dam +
                         pow((pow(ZoB, 2.0) + uRG * ZoB + wRG) -
                                 (dam / (bm * Ru)) * (ZoB - 1.0),
                             2.0) /
                             (pow(pow(ZoB, 2.0) + uRG * ZoB + wRG, 2.0) -
                              (am / (bm * Ru * T)) * (2.0 * ZoB + uRG) *
                                  pow(ZoB - 1.0, 2.0)) -
                         1.0;

          double hDep = Cuw * (am / T - dam) * logZoB + (Z - 1.0);

          // Start h and cp as departure values
          h = Ru * T * hDep / MWmix;
          cp = Ru * cpDep / MWmix;
          {
            const double u = log(T);
            for (int n = 0; n <= ns - 1; n++) {
              // ideal cp/R and h/(RT), Horner in u
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
              // Add departure to individual hi
              hi[n] += Ru * T * hDep / MW(n);
            }
          }

          // Compute density
          rho = p / (Z * Rmix * T);

          // Specific heat ratio
          double dAstardp = Astar / p;
          double dBstardp = Bstar / p;

          dz0 = -(Bstar * dAstardp +
                  (Astar + (2.0 * Bstar + 3.0 * pow(Bstar, 2.0)) * wRG) *
                      dBstardp);
          dz1 = (dAstardp + (2.0 * Bstar * (wRG - uRG) - uRG) * dBstardp);
          dz2 = -(1.0 - uRG) * dBstardp;

          double dZdp = -(dz2 * pow(Z, 2.0) + dz1 * Z + dz0) /
                        (3.0 * pow(Z, 2.0) + 2.0 * Z * z2 + z1);

          double drhodp = (rho / p) * (1.e0 - p * dZdp / Z);
          double drhodt = -(rho / T) * (1.e0 + T * dZdT / Z);
          gamma = drhodp / (drhodp - (T / cp) * pow(drhodt / rho, 2.0));

          // Mixture speed of sound
          c = sqrt(abs(gamma / drhodp));

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

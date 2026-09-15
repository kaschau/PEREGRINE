#include "kernel.hpp"
#include <Kokkos_Core.hpp>

PG_STENCIL(2);

// compiled once per direction: the jit says which, and the index offsets
// fold into the addressing
#ifndef PG_DIRECTION
#error                                                                         \
    "a flux kernel is compiled for one direction: -DPG_DIRECTION from the jit"
#endif
constexpr int iMod = PG_DIRECTION == 0, jMod = PG_DIRECTION == 1,
              kMod = PG_DIRECTION == 2;

PG_RANGE(faces)
PG_ABI void pgFourthOrderKEEP(pgIn *Q_, pgOut *F_, pgIn *A_, pgIn *q_,
                              pgIn *qh_, const pgDims *d, const pgTiling &t) {
#if PG_DIRECTION == 0
  Kokkos::parallel_for(
      "4th Order KEEP i face conv fluxes", t.policy(),
      KOKKOS_LAMBDA(const team &team) {
        const auto r = t.of(team);
        const int e = r.e;
        auto Q = as4(Q_[e]);
        auto F = as4(F_[e]);
        auto A = as4(A_[e]);
        auto q = as4(q_[e]);
        auto qh = as4(qh_[e]);
        const int ni = d[e].ni, nj = d[e].nj, nk = d[e].nk;
        //-------------------------------------------------------------------------------------------|
        // i flux face range
        //-------------------------------------------------------------------------------------------|
        const int order = 4;
        constexpr int half = order / 2;
        constexpr int narray = (half * half + half) / 2;
        const double aq[2] = {2.0 / 3.0, -1.0 / 12.0};
        //-------------------------------------------------------------------------------------------|
        // j flux face range
        //-------------------------------------------------------------------------------------------|
        //-------------------------------------------------------------------------------------------|
        // k flux face range
        //-------------------------------------------------------------------------------------------|
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, r.begin, r.end), [&](const int item) {
              int i, j, k;
              cellAt(t.cells[e], item, i, j, k);

              double U;
              double uf = 0.0;
              double vf = 0.0;
              double wf = 0.0;

              // Reusable arrays
              double uR[narray], vR[narray], wR[narray], rhoR[narray];

              double a;
              // Compute face normal volume flux vector
              double tempu;
              double tempv;
              double tempw;
              int count = 0;
              for (int is = 1; is <= half; is++) {
                tempu = 0.0;
                tempv = 0.0;
                tempw = 0.0;
                for (int js = 0; js <= is - 1; js++) {
                  uR[count] =
                      0.5 * (q(i + js, j, k, 1) + q(i + js - is, j, k, 1));
                  vR[count] =
                      0.5 * (q(i + js, j, k, 2) + q(i + js - is, j, k, 2));
                  wR[count] =
                      0.5 * (q(i + js, j, k, 3) + q(i + js - is, j, k, 3));
                  tempu += uR[count];
                  tempv += vR[count];
                  tempw += wR[count];
                  count++;
                }
                a = aq[is - 1];
                uf += a * tempu;
                vf += a * tempv;
                wf += a * tempw;
              }
              uf *= 2.0;
              vf *= 2.0;
              wf *= 2.0;

              U = A(i, j, k, 0) * uf + A(i, j, k, 1) * vf + A(i, j, k, 2) * wf;

              // Compute fluxes

              // Continuity rho*Ui
              double rho = 0.0;
              double temprho = 0.0;
              count = 0;
              for (int is = 1; is <= half; is++) {
                a = aq[is - 1];
                temprho = 0.0;
                for (int js = 0; js <= is - 1; js++) {
                  rhoR[count] =
                      0.5 * (Q(i + js, j, k, 0) + Q(i + js - is, j, k, 0));
                  temprho += rhoR[count];
                  count++;
                }
                rho += a * temprho;
              }
              rho *= 2.0;

              F(i, j, k, 0) = rho * U;

              // x momentum rho*u*Ui+ p*Ax
              // y momentum rho*v*Ui+ p*Ay
              // w momentum rho*w*Ui+ p*Az
              double rhou = 0.0;
              double rhov = 0.0;
              double rhow = 0.0;
              double p = 0.0;
              double temprhou, temprhov, temprhow, tempp;
              count = 0;
              for (int is = 1; is <= half; is++) {
                a = aq[is - 1];
                temprhou = 0.0;
                temprhov = 0.0;
                temprhow = 0.0;
                tempp = 0.0;
                for (int js = 0; js <= is - 1; js++) {
                  temprhou += rhoR[count] * uR[count];
                  temprhov += rhoR[count] * vR[count];
                  temprhow += rhoR[count] * wR[count];
                  tempp += 0.5 * (q(i + js, j, k, 0) + q(i + js - is, j, k, 0));
                  count++;
                }
                rhou += a * temprhou;
                rhov += a * temprhov;
                rhow += a * temprhow;
                p += a * tempp;
              }
              rhow *= 2.0;
              rhov *= 2.0;
              rhou *= 2.0;
              p *= 2.0;

              F(i, j, k, 1) = rhou * U + p * A(i, j, k, 0);

              F(i, j, k, 2) = rhov * U + p * A(i, j, k, 1);

              F(i, j, k, 3) = rhow * U + p * A(i, j, k, 2);

              // Total energy (rhoE+ p)*Ui)
              double rhoE = 0.0;
              double pu = 0.0;
              double temprhoE, temppu;
              double e, em;
              count = 0;
              for (int is = 1; is <= half; is++) {
                a = aq[is - 1];
                temprhoE = 0.0;
                temppu = 0.0;
                for (int js = 0; js <= is - 1; js++) {
                  e = qh(i + js, j, k, 4) / Q(i + js, j, k, 0);
                  em = qh(i + js - is, j, k, 4) / Q(i + js - is, j, k, 0);

                  temprhoE +=
                      rhoR[count] *
                      (0.5 * (e + em) +
                       0.5 * (q(i + js, j, k, 1) * q(i + js - is, j, k, 1) +
                              q(i + js, j, k, 2) * q(i + js - is, j, k, 2) +
                              q(i + js, j, k, 3) * q(i + js - is, j, k, 3)));
                  count++;

                  temppu +=
                      0.5 * (q(i + js - is, j, k, 0) *
                                 (q(i + js, j, k, 1) * A(i, j, k, 0) +
                                  q(i + js, j, k, 2) * A(i, j, k, 1) +
                                  q(i + js, j, k, 3) * A(i, j, k, 2)) +
                             q(i + js, j, k, 0) *
                                 (q(i + js - is, j, k, 1) * A(i, j, k, 0) +
                                  q(i + js - is, j, k, 2) * A(i, j, k, 1) +
                                  q(i + js - is, j, k, 3) * A(i, j, k, 2)));
                }
                rhoE += a * temprhoE;
                pu += a * temppu;
              }
              rhoE *= 2.0;
              pu *= 2.0;

              F(i, j, k, 4) = rhoE * U + pu;

              // Species
              for (int n = 0; n < ne - 5; n++) {
                double rhoY = 0.0;
                double temprhoY = 0.0;
                count = 0;
                for (int is = 1; is <= order / 2; is++) {
                  a = aq[is - 1];
                  temprhoY = 0.0;
                  for (int js = 0; js <= is - 1; js++) {
                    temprhoY +=
                        rhoR[count] * 0.5 *
                        (q(i + js, j, k, 5 + n) + q(i + js - is, j, k, 5 + n));
                    count++;
                  }
                  rhoY += a * temprhoY;
                }
                rhoY *= 2.0;
                F(i, j, k, 5 + n) = rhoY * U;
              }
            });
      });
#endif
#if PG_DIRECTION == 1
  Kokkos::parallel_for(
      "4th Order KEEP j face conv fluxes", t.policy(),
      KOKKOS_LAMBDA(const team &team) {
        const auto r = t.of(team);
        const int e = r.e;
        auto Q = as4(Q_[e]);
        auto F = as4(F_[e]);
        auto A = as4(A_[e]);
        auto q = as4(q_[e]);
        auto qh = as4(qh_[e]);
        const int ni = d[e].ni, nj = d[e].nj, nk = d[e].nk;
        //-------------------------------------------------------------------------------------------|
        // i flux face range
        //-------------------------------------------------------------------------------------------|
        const int order = 4;
        constexpr int half = order / 2;
        constexpr int narray = (half * half + half) / 2;
        const double aq[2] = {2.0 / 3.0, -1.0 / 12.0};
        //-------------------------------------------------------------------------------------------|
        // j flux face range
        //-------------------------------------------------------------------------------------------|
        //-------------------------------------------------------------------------------------------|
        // k flux face range
        //-------------------------------------------------------------------------------------------|
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, r.begin, r.end), [&](const int item) {
              int i, j, k;
              cellAt(t.cells[e], item, i, j, k);

              double V;
              double uf = 0.0;
              double vf = 0.0;
              double wf = 0.0;

              // Reusable arrays
              double uR[narray], vR[narray], wR[narray], rhoR[narray];

              double a;
              // Compute face normal volume flux vector
              double tempu;
              double tempv;
              double tempw;
              int count = 0;
              for (int is = 1; is <= half; is++) {
                tempu = 0.0;
                tempv = 0.0;
                tempw = 0.0;
                for (int js = 0; js <= is - 1; js++) {
                  uR[count] =
                      0.5 * (q(i, j + js, k, 1) + q(i, j + js - is, k, 1));
                  vR[count] =
                      0.5 * (q(i, j + js, k, 2) + q(i, j + js - is, k, 2));
                  wR[count] =
                      0.5 * (q(i, j + js, k, 3) + q(i, j + js - is, k, 3));
                  tempu += uR[count];
                  tempv += vR[count];
                  tempw += wR[count];
                  count++;
                }
                a = aq[is - 1];
                uf += a * tempu;
                vf += a * tempv;
                wf += a * tempw;
              }
              uf *= 2.0;
              vf *= 2.0;
              wf *= 2.0;

              V = A(i, j, k, 0) * uf + A(i, j, k, 1) * vf + A(i, j, k, 2) * wf;

              // Compute fluxes

              // Continuity rho*Vi
              double rho = 0.0;
              double temprho = 0.0;
              count = 0;
              for (int is = 1; is <= half; is++) {
                a = aq[is - 1];
                temprho = 0.0;
                for (int js = 0; js <= is - 1; js++) {
                  rhoR[count] =
                      0.5 * (Q(i, j + js, k, 0) + Q(i, j + js - is, k, 0));
                  temprho += rhoR[count];
                  count++;
                }
                rho += a * temprho;
              }
              rho *= 2.0;

              F(i, j, k, 0) = rho * V;

              // x momentum rho*u*Vi+ p*Ax
              // y momentum rho*v*Vi+ p*Ay
              // w momentum rho*w*Vi+ p*Az
              double rhou = 0.0;
              double rhov = 0.0;
              double rhow = 0.0;
              double p = 0.0;
              double temprhou, temprhov, temprhow, tempp;
              count = 0;
              for (int is = 1; is <= half; is++) {
                a = aq[is - 1];
                temprhou = 0.0;
                temprhov = 0.0;
                temprhow = 0.0;
                tempp = 0.0;
                for (int js = 0; js <= is - 1; js++) {
                  temprhou += rhoR[count] * uR[count];
                  temprhov += rhoR[count] * vR[count];
                  temprhow += rhoR[count] * wR[count];
                  tempp += 0.5 * (q(i, j + js, k, 0) + q(i, j + js - is, k, 0));
                  count++;
                }
                rhou += a * temprhou;
                rhov += a * temprhov;
                rhow += a * temprhow;
                p += a * tempp;
              }
              rhow *= 2.0;
              rhov *= 2.0;
              rhou *= 2.0;
              p *= 2.0;

              F(i, j, k, 1) = rhou * V + p * A(i, j, k, 0);

              F(i, j, k, 2) = rhov * V + p * A(i, j, k, 1);

              F(i, j, k, 3) = rhow * V + p * A(i, j, k, 2);

              // Total energy (rhoE+ p)*Vi)
              double rhoE = 0.0;
              double pu = 0.0;
              double temprhoE, temppu;
              double e, em;
              count = 0;
              for (int is = 1; is <= half; is++) {
                a = aq[is - 1];
                temprhoE = 0.0;
                temppu = 0.0;
                for (int js = 0; js <= is - 1; js++) {
                  e = qh(i, j + js, k, 4) / Q(i, j + js, k, 0);
                  em = qh(i, j + js - is, k, 4) / Q(i, j + js - is, k, 0);

                  temprhoE +=
                      rhoR[count] *
                      (0.5 * (e + em) +
                       0.5 * (q(i, j + js, k, 1) * q(i, j + js - is, k, 1) +
                              q(i, j + js, k, 2) * q(i, j + js - is, k, 2) +
                              q(i, j + js, k, 3) * q(i, j + js - is, k, 3)));
                  count++;

                  temppu +=
                      0.5 * (q(i, j + js - is, k, 0) *
                                 (q(i, j + js, k, 1) * A(i, j, k, 0) +
                                  q(i, j + js, k, 2) * A(i, j, k, 1) +
                                  q(i, j + js, k, 3) * A(i, j, k, 2)) +
                             q(i, j + js, k, 0) *
                                 (q(i, j + js - is, k, 1) * A(i, j, k, 0) +
                                  q(i, j + js - is, k, 2) * A(i, j, k, 1) +
                                  q(i, j + js - is, k, 3) * A(i, j, k, 2)));
                }
                rhoE += a * temprhoE;
                pu += a * temppu;
              }
              rhoE *= 2.0;
              pu *= 2.0;

              F(i, j, k, 4) = rhoE * V + pu;

              // Species
              for (int n = 0; n < ne - 5; n++) {
                double rhoY = 0.0;
                double temprhoY = 0.0;
                count = 0;
                for (int is = 1; is <= order / 2; is++) {
                  a = aq[is - 1];
                  temprhoY = 0.0;
                  for (int js = 0; js <= is - 1; js++) {
                    temprhoY +=
                        rhoR[count] * 0.5 *
                        (q(i, j + js, k, 5 + n) + q(i, j + js - is, k, 5 + n));
                    count++;
                  }
                  rhoY += a * temprhoY;
                }
                rhoY *= 2.0;
                F(i, j, k, 5 + n) = rhoY * V;
              }
            });
      });
#endif
#if PG_DIRECTION == 2
  Kokkos::parallel_for(
      "4th Order KEEP k face conv fluxes", t.policy(),
      KOKKOS_LAMBDA(const team &team) {
        const auto r = t.of(team);
        const int e = r.e;
        auto Q = as4(Q_[e]);
        auto F = as4(F_[e]);
        auto A = as4(A_[e]);
        auto q = as4(q_[e]);
        auto qh = as4(qh_[e]);
        const int ni = d[e].ni, nj = d[e].nj, nk = d[e].nk;
        //-------------------------------------------------------------------------------------------|
        // i flux face range
        //-------------------------------------------------------------------------------------------|
        const int order = 4;
        constexpr int half = order / 2;
        constexpr int narray = (half * half + half) / 2;
        const double aq[2] = {2.0 / 3.0, -1.0 / 12.0};
        //-------------------------------------------------------------------------------------------|
        // j flux face range
        //-------------------------------------------------------------------------------------------|
        //-------------------------------------------------------------------------------------------|
        // k flux face range
        //-------------------------------------------------------------------------------------------|
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, r.begin, r.end), [&](const int item) {
              int i, j, k;
              cellAt(t.cells[e], item, i, j, k);

              double W;
              double uf = 0.0;
              double vf = 0.0;
              double wf = 0.0;

              // Reusable arrays
              double uR[narray], vR[narray], wR[narray], rhoR[narray];

              double a;
              // Compute face normal volume flux vector
              double tempu;
              double tempv;
              double tempw;
              int count = 0;
              for (int is = 1; is <= half; is++) {
                tempu = 0.0;
                tempv = 0.0;
                tempw = 0.0;
                for (int js = 0; js <= is - 1; js++) {
                  uR[count] =
                      0.5 * (q(i, j, k + js, 1) + q(i, j, k + js - is, 1));
                  vR[count] =
                      0.5 * (q(i, j, k + js, 2) + q(i, j, k + js - is, 2));
                  wR[count] =
                      0.5 * (q(i, j, k + js, 3) + q(i, j, k + js - is, 3));
                  tempu += uR[count];
                  tempv += vR[count];
                  tempw += wR[count];
                  count++;
                }
                a = aq[is - 1];
                uf += a * tempu;
                vf += a * tempv;
                wf += a * tempw;
              }
              uf *= 2.0;
              vf *= 2.0;
              wf *= 2.0;

              W = A(i, j, k, 0) * uf + A(i, j, k, 1) * vf + A(i, j, k, 2) * wf;

              // Compute fluxes

              // Continuity rho*Wi
              double rho = 0.0;
              double temprho = 0.0;
              count = 0;
              for (int is = 1; is <= half; is++) {
                a = aq[is - 1];
                temprho = 0.0;
                for (int js = 0; js <= is - 1; js++) {
                  rhoR[count] =
                      0.5 * (Q(i, j, k + js, 0) + Q(i, j, k + js - is, 0));
                  temprho += rhoR[count];
                  count++;
                }
                rho += a * temprho;
              }
              rho *= 2.0;

              F(i, j, k, 0) = rho * W;

              // x momentum rho*u*Wi+ p*Ax
              // y momentum rho*v*Wi+ p*Ay
              // w momentum rho*w*Wi+ p*Az
              double rhou = 0.0;
              double rhov = 0.0;
              double rhow = 0.0;
              double p = 0.0;
              double temprhou, temprhov, temprhow, tempp;
              count = 0;
              for (int is = 1; is <= half; is++) {
                a = aq[is - 1];
                temprhou = 0.0;
                temprhov = 0.0;
                temprhow = 0.0;
                tempp = 0.0;
                for (int js = 0; js <= is - 1; js++) {
                  temprhou += rhoR[count] * uR[count];
                  temprhov += rhoR[count] * vR[count];
                  temprhow += rhoR[count] * wR[count];
                  tempp += 0.5 * (q(i, j, k + js, 0) + q(i, j, k + js - is, 0));
                  count++;
                }
                rhou += a * temprhou;
                rhov += a * temprhov;
                rhow += a * temprhow;
                p += a * tempp;
              }
              rhow *= 2.0;
              rhov *= 2.0;
              rhou *= 2.0;
              p *= 2.0;

              F(i, j, k, 1) = rhou * W + p * A(i, j, k, 0);

              F(i, j, k, 2) = rhov * W + p * A(i, j, k, 1);

              F(i, j, k, 3) = rhow * W + p * A(i, j, k, 2);

              // Total energy (rhoE+ p)*Wi)
              double rhoE = 0.0;
              double pu = 0.0;
              double temprhoE, temppu;
              double e, em;
              count = 0;
              for (int is = 1; is <= half; is++) {
                a = aq[is - 1];
                temprhoE = 0.0;
                temppu = 0.0;
                for (int js = 0; js <= is - 1; js++) {
                  e = qh(i, j, k + js, 4) / Q(i, j, k + js, 0);
                  em = qh(i, j, k + js - is, 4) / Q(i, j, k + js - is, 0);

                  temprhoE +=
                      rhoR[count] *
                      (0.5 * (e + em) +
                       0.5 * (q(i, j, k + js, 1) * q(i, j, k + js - is, 1) +
                              q(i, j, k + js, 2) * q(i, j, k + js - is, 2) +
                              q(i, j, k + js, 3) * q(i, j, k + js - is, 3)));
                  count++;

                  temppu +=
                      0.5 * (q(i, j, k + js - is, 0) *
                                 (q(i, j, k + js, 1) * A(i, j, k, 0) +
                                  q(i, j, k + js, 2) * A(i, j, k, 1) +
                                  q(i, j, k + js, 3) * A(i, j, k, 2)) +
                             q(i, j, k + js, 0) *
                                 (q(i, j, k + js - is, 1) * A(i, j, k, 0) +
                                  q(i, j, k + js - is, 2) * A(i, j, k, 1) +
                                  q(i, j, k + js - is, 3) * A(i, j, k, 2)));
                }
                rhoE += a * temprhoE;
                pu += a * temppu;
              }
              rhoE *= 2.0;
              pu *= 2.0;

              F(i, j, k, 4) = rhoE * W + pu;

              // Species
              for (int n = 0; n < ne - 5; n++) {
                double rhoY = 0.0;
                double temprhoY = 0.0;
                count = 0;
                for (int is = 1; is <= order / 2; is++) {
                  a = aq[is - 1];
                  temprhoY = 0.0;
                  for (int js = 0; js <= is - 1; js++) {
                    temprhoY +=
                        rhoR[count] * 0.5 *
                        (q(i, j, k + js, 5 + n) + q(i, j, k + js - is, 5 + n));
                    count++;
                  }
                  rhoY += a * temprhoY;
                }
                rhoY *= 2.0;
                F(i, j, k, 5 + n) = rhoY * W;
              }
            });
      });
#endif
}

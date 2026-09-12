#include "abi.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>

PG_ABI void pgFourthOrderKEEP(const pgView *Q_, const pgView *iF_,
                              const pgView *iS_, const pgView *jF_,
                              const pgView *jS_, const pgView *kF_,
                              const pgView *kS_, const pgView *q_,
                              const pgView *qh_, const pgDims *d) {
  auto Q = as4(*Q_);
  auto iF = as4(*iF_);
  auto iS = as4(*iS_);
  auto jF = as4(*jF_);
  auto jS = as4(*jS_);
  auto kF = as4(*kF_);
  auto kS = as4(*kS_);
  auto q = as4(*q_);
  auto qh = as4(*qh_);
  const int ng = d->ng, ni = d->ni, nj = d->nj, nk = d->nk;
  const int ne = Q.extent(3);

  //-------------------------------------------------------------------------------------------|
  // i flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_i({ng, ng, ng}, {ni + ng, nj + ng - 1, nk + ng - 1});

  const int order = 4;
  constexpr int half = order / 2;
  constexpr int narray = (half * half + half) / 2;
  const double aq[2] = {2.0 / 3.0, -1.0 / 12.0};

  Kokkos::parallel_for(
      "4th Order KEEP i face conv fluxes", range_i,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
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
            uR[count] = 0.5 * (q(i + js, j, k, 1) + q(i + js - is, j, k, 1));
            vR[count] = 0.5 * (q(i + js, j, k, 2) + q(i + js - is, j, k, 2));
            wR[count] = 0.5 * (q(i + js, j, k, 3) + q(i + js - is, j, k, 3));
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

        U = iS(i, j, k, 0) * uf + iS(i, j, k, 1) * vf + iS(i, j, k, 2) * wf;

        // Compute fluxes

        // Continuity rho*Ui
        double rho = 0.0;
        double temprho = 0.0;
        count = 0;
        for (int is = 1; is <= half; is++) {
          a = aq[is - 1];
          temprho = 0.0;
          for (int js = 0; js <= is - 1; js++) {
            rhoR[count] = 0.5 * (Q(i + js, j, k, 0) + Q(i + js - is, j, k, 0));
            temprho += rhoR[count];
            count++;
          }
          rho += a * temprho;
        }
        rho *= 2.0;

        iF(i, j, k, 0) = rho * U;

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

        iF(i, j, k, 1) = rhou * U + p * iS(i, j, k, 0);

        iF(i, j, k, 2) = rhov * U + p * iS(i, j, k, 1);

        iF(i, j, k, 3) = rhow * U + p * iS(i, j, k, 2);

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

            temprhoE += rhoR[count] *
                        (0.5 * (e + em) +
                         0.5 * (q(i + js, j, k, 1) * q(i + js - is, j, k, 1) +
                                q(i + js, j, k, 2) * q(i + js - is, j, k, 2) +
                                q(i + js, j, k, 3) * q(i + js - is, j, k, 3)));
            count++;

            temppu += 0.5 * (q(i + js - is, j, k, 0) *
                                 (q(i + js, j, k, 1) * iS(i, j, k, 0) +
                                  q(i + js, j, k, 2) * iS(i, j, k, 1) +
                                  q(i + js, j, k, 3) * iS(i, j, k, 2)) +
                             q(i + js, j, k, 0) *
                                 (q(i + js - is, j, k, 1) * iS(i, j, k, 0) +
                                  q(i + js - is, j, k, 2) * iS(i, j, k, 1) +
                                  q(i + js - is, j, k, 3) * iS(i, j, k, 2)));
          }
          rhoE += a * temprhoE;
          pu += a * temppu;
        }
        rhoE *= 2.0;
        pu *= 2.0;

        iF(i, j, k, 4) = rhoE * U + pu;

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
          iF(i, j, k, 5 + n) = rhoY * U;
        }
      });

  //-------------------------------------------------------------------------------------------|
  // j flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_j({ng, ng, ng}, {ni + ng - 1, nj + ng, nk + ng - 1});
  Kokkos::parallel_for(
      "4th Order KEEP j face conv fluxes", range_j,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
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
            uR[count] = 0.5 * (q(i, j + js, k, 1) + q(i, j + js - is, k, 1));
            vR[count] = 0.5 * (q(i, j + js, k, 2) + q(i, j + js - is, k, 2));
            wR[count] = 0.5 * (q(i, j + js, k, 3) + q(i, j + js - is, k, 3));
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

        V = jS(i, j, k, 0) * uf + jS(i, j, k, 1) * vf + jS(i, j, k, 2) * wf;

        // Compute fluxes

        // Continuity rho*Vi
        double rho = 0.0;
        double temprho = 0.0;
        count = 0;
        for (int is = 1; is <= half; is++) {
          a = aq[is - 1];
          temprho = 0.0;
          for (int js = 0; js <= is - 1; js++) {
            rhoR[count] = 0.5 * (Q(i, j + js, k, 0) + Q(i, j + js - is, k, 0));
            temprho += rhoR[count];
            count++;
          }
          rho += a * temprho;
        }
        rho *= 2.0;

        jF(i, j, k, 0) = rho * V;

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

        jF(i, j, k, 1) = rhou * V + p * jS(i, j, k, 0);

        jF(i, j, k, 2) = rhov * V + p * jS(i, j, k, 1);

        jF(i, j, k, 3) = rhow * V + p * jS(i, j, k, 2);

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

            temprhoE += rhoR[count] *
                        (0.5 * (e + em) +
                         0.5 * (q(i, j + js, k, 1) * q(i, j + js - is, k, 1) +
                                q(i, j + js, k, 2) * q(i, j + js - is, k, 2) +
                                q(i, j + js, k, 3) * q(i, j + js - is, k, 3)));
            count++;

            temppu += 0.5 * (q(i, j + js - is, k, 0) *
                                 (q(i, j + js, k, 1) * jS(i, j, k, 0) +
                                  q(i, j + js, k, 2) * jS(i, j, k, 1) +
                                  q(i, j + js, k, 3) * jS(i, j, k, 2)) +
                             q(i, j + js, k, 0) *
                                 (q(i, j + js - is, k, 1) * jS(i, j, k, 0) +
                                  q(i, j + js - is, k, 2) * jS(i, j, k, 1) +
                                  q(i, j + js - is, k, 3) * jS(i, j, k, 2)));
          }
          rhoE += a * temprhoE;
          pu += a * temppu;
        }
        rhoE *= 2.0;
        pu *= 2.0;

        jF(i, j, k, 4) = rhoE * V + pu;

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
          jF(i, j, k, 5 + n) = rhoY * V;
        }
      });

  //-------------------------------------------------------------------------------------------|
  // k flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_k({ng, ng, ng}, {ni + ng - 1, nj + ng - 1, nk + ng});
  Kokkos::parallel_for(
      "4th Order KEEP k face conv fluxes", range_k,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
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
            uR[count] = 0.5 * (q(i, j, k + js, 1) + q(i, j, k + js - is, 1));
            vR[count] = 0.5 * (q(i, j, k + js, 2) + q(i, j, k + js - is, 2));
            wR[count] = 0.5 * (q(i, j, k + js, 3) + q(i, j, k + js - is, 3));
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

        W = kS(i, j, k, 0) * uf + kS(i, j, k, 1) * vf + kS(i, j, k, 2) * wf;

        // Compute fluxes

        // Continuity rho*Wi
        double rho = 0.0;
        double temprho = 0.0;
        count = 0;
        for (int is = 1; is <= half; is++) {
          a = aq[is - 1];
          temprho = 0.0;
          for (int js = 0; js <= is - 1; js++) {
            rhoR[count] = 0.5 * (Q(i, j, k + js, 0) + Q(i, j, k + js - is, 0));
            temprho += rhoR[count];
            count++;
          }
          rho += a * temprho;
        }
        rho *= 2.0;

        kF(i, j, k, 0) = rho * W;

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

        kF(i, j, k, 1) = rhou * W + p * kS(i, j, k, 0);

        kF(i, j, k, 2) = rhov * W + p * kS(i, j, k, 1);

        kF(i, j, k, 3) = rhow * W + p * kS(i, j, k, 2);

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

            temprhoE += rhoR[count] *
                        (0.5 * (e + em) +
                         0.5 * (q(i, j, k + js, 1) * q(i, j, k + js - is, 1) +
                                q(i, j, k + js, 2) * q(i, j, k + js - is, 2) +
                                q(i, j, k + js, 3) * q(i, j, k + js - is, 3)));
            count++;

            temppu += 0.5 * (q(i, j, k + js - is, 0) *
                                 (q(i, j, k + js, 1) * kS(i, j, k, 0) +
                                  q(i, j, k + js, 2) * kS(i, j, k, 1) +
                                  q(i, j, k + js, 3) * kS(i, j, k, 2)) +
                             q(i, j, k + js, 0) *
                                 (q(i, j, k + js - is, 1) * kS(i, j, k, 0) +
                                  q(i, j, k + js - is, 2) * kS(i, j, k, 1) +
                                  q(i, j, k + js - is, 3) * kS(i, j, k, 2)));
          }
          rhoE += a * temprhoE;
          pu += a * temppu;
        }
        rhoE *= 2.0;
        pu *= 2.0;

        kF(i, j, k, 4) = rhoE * W + pu;

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
          kF(i, j, k, 5 + n) = rhoY * W;
        }
      });
}

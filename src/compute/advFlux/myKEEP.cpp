#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"

void myKEEP(block_ &b) {

  //-------------------------------------------------------------------------------------------|
  // i flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_i({b.ng, b.ng, b.ng},
                   {b.ni + b.ng, b.nj + b.ng - 1, b.nk + b.ng - 1});
  Kokkos::parallel_for(
      "2nd order KEEP i face conv fluxes", range_i,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double U;
        double uf;
        double vf;
        double wf;
        double pf;

        // Compute face normal volume flux vector
        uf = 0.5 * (b.q(i, j, k, 1) + b.q(i - 1, j, k, 1));
        vf = 0.5 * (b.q(i, j, k, 2) + b.q(i - 1, j, k, 2));
        wf = 0.5 * (b.q(i, j, k, 3) + b.q(i - 1, j, k, 3));

        U = b.isx(i, j, k) * uf + b.isy(i, j, k) * vf + b.isz(i, j, k) * wf;

        pf = 0.5 * (b.q(i, j, k, 0) + b.q(i - 1, j, k, 0));

        // Compute fluxes
        double rho;
        rho = 0.5 * (b.Q(i, j, k, 0) + b.Q(i - 1, j, k, 0));

        // Continuity rho*Ui
        b.iF(i, j, k, 0) = rho * U;

        // x momentum rho*u*Ui+ p*Ax
        b.iF(i, j, k, 1) = rho * uf * U + pf * b.isx(i, j, k);

        // y momentum rho*v*Ui+ p*Ay
        b.iF(i, j, k, 2) = rho * vf * U + pf * b.isy(i, j, k);

        // w momentum rho*w*Ui+ p*Az
        b.iF(i, j, k, 3) = rho * wf * U + pf * b.isz(i, j, k);

        // Total energy (rhoE+ p)*Ui)
        double Kj = rho * 0.5 *
                    (b.q(i, j, k, 1) * b.q(i - 1, j, k, 1) +
                     b.q(i, j, k, 2) * b.q(i - 1, j, k, 2) +
                     b.q(i, j, k, 3) * b.q(i - 1, j, k, 3)) *
                    U;

        double Pj =
            0.5 * (b.q(i - 1, j, k, 0) * (b.q(i, j, k, 1) * b.isx(i, j, k) +
                                          b.q(i, j, k, 2) * b.isy(i, j, k) +
                                          b.q(i, j, k, 3) * b.isz(i, j, k)) +
                   b.q(i, j, k, 0) * (b.q(i - 1, j, k, 1) * b.isx(i, j, k) +
                                      b.q(i - 1, j, k, 2) * b.isy(i, j, k) +
                                      b.q(i - 1, j, k, 3) * b.isz(i, j, k)));

        // solve for internal energy flux
        double cvR = b.qh(i, j, k, 1) / b.qh(i, j, k, 0);
        double &TR = b.q(i, j, k, 4);
        double &pR = b.q(i, j, k, 0);
        double &rhoR = b.Q(i, j, k, 0);
        double RR = b.qh(i, j, k, 1) - cvR;
        double hR = b.qh(i, j, k, 2) / rhoR;
        double &uR = b.q(i, j, k, 1);
        double &vR = b.q(i, j, k, 2);
        double &wR = b.q(i, j, k, 3);
        double sR = cvR * log(TR) - RR * log(rhoR);

        double v0R =
            sR + (-hR + 0.5 * (pow(uR, 2) + pow(vR, 2) + pow(wR, 2))) / TR;
        double v1R = -uR / TR;
        double v2R = -vR / TR;
        double v3R = -wR / TR;
        double v4R = 1.0 / TR;

        // left
        double cvL = b.qh(i - 1, j, k, 1) / b.qh(i - 1, j, k, 0);
        double &TL = b.q(i - 1, j, k, 4);
        double &pL = b.q(i - 1, j, k, 0);
        double &rhoL = b.Q(i - 1, j, k, 0);
        double RL = b.qh(i - 1, j, k, 1) - cvL;
        double hL = b.qh(i - 1, j, k, 2) / rhoL;
        double &uL = b.q(i - 1, j, k, 1);
        double &vL = b.q(i - 1, j, k, 2);
        double &wL = b.q(i - 1, j, k, 3);
        double sL = cvL * log(TL) - RL * log(rhoL);

        double v0L =
            sL + (-hL + 0.5 * (pow(uL, 2) + pow(vL, 2) + pow(wL, 2))) / TL;
        double v1L = -uL / TL;
        double v2L = -vL / TL;
        double v3L = -wL / TL;
        double v4L = 1.0 / TL;

        double Fs = rho * U * 0.5 * (sR + sL);

        double phiR = -pR * uR / TR * b.iS(i, j, k);
        double phiL = -pL * uL / TL * b.iS(i, j, k);

        double PHI = 0.5 * (phiR + phiL);
        double V0 = 0.5 * (v0R + v0L);
        double V1 = 0.5 * (v1R + v1L);
        double V2 = 0.5 * (v2R + v2L);
        double V3 = 0.5 * (v3R + v3L);
        double V4 = 0.5 * (v4R + v4L);

        double Ij = (Fs + PHI - V0 * b.iF(i, j, k, 0) - V1 * b.iF(i, j, k, 1) -
                     V2 * b.iF(i, j, k, 2) - V3 * b.iF(i, j, k, 3)) /
                        V4 -
                    Pj - Kj;

        // double &Cj = b.iF(i, j, k, 0);
        // double Ij =
        //     (0.5 * (sm - s) * T + h) * Cj +
        //     0.5 * (b.q(i, j, k, 0) * ((b.q(i, j, k, 1) * b.isx(i, j, k) +
        //                                b.q(i, j, k, 2) * b.isy(i, j, k) +
        //                                b.q(i, j, k, 3) * b.isz(i, j, k)) -
        //                               (b.q(i - 1, j, k, 1) * b.isx(i, j, k) +
        //                                b.q(i - 1, j, k, 2) * b.isy(i, j, k) +
        //                                b.q(i - 1, j, k, 3) * b.isz(i, j,
        //                                k))));

        b.iF(i, j, k, 4) = Ij + Kj + Pj;

        // Species
        for (int n = 0; n < b.ne - 5; n++) {
          b.iF(i, j, k, 5 + n) =
              0.5 * (b.Q(i, j, k, 5 + n) + b.Q(i - 1, j, k, 5 + n)) * U;
        }
      });

  //-------------------------------------------------------------------------------------------|
  // j flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_j({b.ng, b.ng, b.ng},
                   {b.ni + b.ng - 1, b.nj + b.ng, b.nk + b.ng - 1});
  Kokkos::parallel_for(
      "2nd order KEEP j face conv fluxes", range_j,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double V;
        double uf;
        double vf;
        double wf;
        double pf;

        // Compute face normal volume flux vector
        uf = 0.5 * (b.q(i, j, k, 1) + b.q(i, j - 1, k, 1));
        vf = 0.5 * (b.q(i, j, k, 2) + b.q(i, j - 1, k, 2));
        wf = 0.5 * (b.q(i, j, k, 3) + b.q(i, j - 1, k, 3));

        V = b.jsx(i, j, k) * uf + b.jsy(i, j, k) * vf + b.jsz(i, j, k) * wf;

        pf = 0.5 * (b.q(i, j, k, 0) + b.q(i, j - 1, k, 0));

        // Compute fluxes
        double rho;
        rho = 0.5 * (b.Q(i, j, k, 0) + b.Q(i, j - 1, k, 0));

        // Continuity rho*Vj
        b.jF(i, j, k, 0) = rho * V;

        // x momentum rho*u*Vj+ pAx
        b.jF(i, j, k, 1) = rho * uf * V + pf * b.jsx(i, j, k);

        // y momentum rho*v*Vj+ pAy
        b.jF(i, j, k, 2) = rho * vf * V + pf * b.jsy(i, j, k);

        // w momentum rho*w*Vj+ pAz
        b.jF(i, j, k, 3) = rho * wf * V + pf * b.jsz(i, j, k);

        // Total energy (rhoE+P)*Vj)
        double e;
        double em;

        e = b.qh(i, j, k, 4);
        em = b.qh(i, j - 1, k, 4);

        b.jF(i, j, k, 4) =
            ((0.5 * (e + em) + rho * 0.5 *
                                   (b.q(i, j, k, 1) * b.q(i, j - 1, k, 1) +
                                    b.q(i, j, k, 2) * b.q(i, j - 1, k, 2) +
                                    b.q(i, j, k, 3) * b.q(i, j - 1, k, 3)))) *
            V;

        b.jF(i, j, k, 4) +=
            0.5 * (b.q(i, j - 1, k, 0) * (b.q(i, j, k, 1) * b.jsx(i, j, k) +
                                          b.q(i, j, k, 2) * b.jsy(i, j, k) +
                                          b.q(i, j, k, 3) * b.jsz(i, j, k)) +
                   b.q(i, j, k, 0) * (b.q(i, j - 1, k, 1) * b.jsx(i, j, k) +
                                      b.q(i, j - 1, k, 2) * b.jsy(i, j, k) +
                                      b.q(i, j - 1, k, 3) * b.jsz(i, j, k)));
        // Species
        for (int n = 0; n < b.ne - 5; n++) {
          b.jF(i, j, k, 5 + n) =
              0.5 * (b.Q(i, j, k, 5 + n) + b.Q(i, j - 1, k, 5 + n)) * V;
        }
      });

  //-------------------------------------------------------------------------------------------|
  // k flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_k({b.ng, b.ng, b.ng},
                   {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng});
  Kokkos::parallel_for(
      "2nd order KEEP k face conv fluxes", range_k,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double W;
        double uf;
        double vf;
        double wf;
        double pf;

        // Compute face normal volume flux vector
        uf = 0.5 * (b.q(i, j, k, 1) + b.q(i, j, k - 1, 1));
        vf = 0.5 * (b.q(i, j, k, 2) + b.q(i, j, k - 1, 2));
        wf = 0.5 * (b.q(i, j, k, 3) + b.q(i, j, k - 1, 3));

        W = b.ksx(i, j, k) * uf + b.ksy(i, j, k) * vf + b.ksz(i, j, k) * wf;

        pf = 0.5 * (b.q(i, j, k, 0) + b.q(i, j, k - 1, 0));

        // Compute fluxes
        double rho;
        rho = 0.5 * (b.Q(i, j, k, 0) + b.Q(i, j, k - 1, 0));
        // Continuity rho*Wk
        b.kF(i, j, k, 0) = rho * W;

        // x momentum rho*u*Wk+ pAx
        b.kF(i, j, k, 1) = rho * uf * W + pf * b.ksx(i, j, k);

        // y momentum rho*v*Wk+ pAy
        b.kF(i, j, k, 2) = rho * vf * W + pf * b.ksy(i, j, k);

        // w momentum rho*w*Wk+ pAz
        b.kF(i, j, k, 3) = rho * wf * W + pf * b.ksz(i, j, k);

        // Total energy (rhoE+P)*Wk)
        double e;
        double em;

        e = b.qh(i, j, k, 4);
        em = b.qh(i, j, k - 1, 4);

        b.kF(i, j, k, 4) =
            ((0.5 * (e + em) + rho * 0.5 *
                                   (b.q(i, j, k, 1) * b.q(i, j, k - 1, 1) +
                                    b.q(i, j, k, 2) * b.q(i, j, k - 1, 2) +
                                    b.q(i, j, k, 3) * b.q(i, j, k - 1, 3)))) *
            W;

        b.kF(i, j, k, 4) +=
            0.5 * (b.q(i, j, k - 1, 0) * (b.q(i, j, k, 1) * b.ksx(i, j, k) +
                                          b.q(i, j, k, 2) * b.ksy(i, j, k) +
                                          b.q(i, j, k, 3) * b.ksz(i, j, k)) +
                   b.q(i, j, k, 0) * (b.q(i, j, k - 1, 1) * b.ksx(i, j, k) +
                                      b.q(i, j, k - 1, 2) * b.ksy(i, j, k) +
                                      b.q(i, j, k - 1, 3) * b.ksz(i, j, k)));
        // Species
        for (int n = 0; n < b.ne - 5; n++) {
          b.kF(i, j, k, 5 + n) =
              0.5 * (b.Q(i, j, k, 5 + n) + b.Q(i, j, k - 1, 5 + n)) * W;
        }
      });
}

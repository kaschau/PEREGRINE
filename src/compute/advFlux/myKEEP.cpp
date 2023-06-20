#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"

#ifdef NSCOMPILE
#define ns NS
#endif

void myKEEP(block_ &b, const thtrdat_ &th) {

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
        double gk[ns];
        double YR[ns];
        double YL[ns];
        // Compute nth species Y
        YR[ns - 1] = 1.0;
        YL[ns - 1] = 1.0;
        for (int n = 0; n < ns - 1; n++) {
          YR[n] = b.q(i, j, k, 5 + n);
          YL[n] = b.q(i - 1, j, k, 5 + n);
          YR[ns - 1] -= YR[n];
          YL[ns - 1] -= YL[n];
        }

        // right state
        double cvR = b.qh(i, j, k, 1) / b.qh(i, j, k, 0);
        double &TR = b.q(i, j, k, 4);
        double &rhoR = b.Q(i, j, k, 0);
        double RR = b.qh(i, j, k, 1) - cvR;
        double &ur = b.q(i, j, k, 1);
        double &vr = b.q(i, j, k, 2);
        double &wr = b.q(i, j, k, 3);
        double phiR =
            -RR * rhoR *
            (ur * b.isx(i, j, k) + vr * b.isy(i, j, k) + wr * b.isz(i, j, k));

        double skR[ns];
        for (int n = 0; n < ns; n++) {
          if (YR[n] == 0.0) {
            skR[n] = 0.0;
            gk[n] = 0.0;
          } else {
            double cpk = th.cp0(n);
            double Rk = th.Ru / th.MW(n);
            double cvk = cpk - Rk;
            skR[n] = cvk * log(TR) - Rk * log(rhoR * YR[n]);
            double hk = b.qh(i, j, k, 5 + n);
            gk[n] = hk - skR[n] * TR;
          }
        }

        double vR[b.ne];
        vR[0] =
            (-gk[ns - 1] + 0.5 * (pow(ur, 2) + pow(vr, 2) + pow(wr, 2))) / TR;
        vR[1] = -ur / TR;
        vR[2] = -vr / TR;
        vR[3] = -wr / TR;
        vR[4] = 1.0 / TR;
        for (int n = 0; n < ns - 1; n++) {
          vR[5 + n] = -(gk[n] - gk[ns - 1]) / TR;
        }

        // left
        double cvL = b.qh(i - 1, j, k, 1) / b.qh(i - 1, j, k, 0);
        double &TL = b.q(i - 1, j, k, 4);
        double &rhoL = b.Q(i - 1, j, k, 0);
        double RL = b.qh(i - 1, j, k, 1) - cvL;
        double &ul = b.q(i - 1, j, k, 1);
        double &vl = b.q(i - 1, j, k, 2);
        double &wl = b.q(i - 1, j, k, 3);
        double phiL =
            -RL * rhoL *
            (ul * b.isx(i, j, k) + vl * b.isy(i, j, k) + wl * b.isz(i, j, k));

        double skL[ns];
        for (int n = 0; n < ns; n++) {
          if (YL[n] == 0.0) {
            skL[n] = 0.0;
            gk[n] = 0.0;
          } else {
            double cpk = th.cp0(n);
            double Rk = th.Ru / th.MW(n);
            double cvk = cpk - Rk;
            skL[n] = cvk * log(TL) - Rk * log(rhoL * YL[n]);
            double hk = b.qh(i - 1, j, k, 5 + n);
            gk[n] = hk - skL[n] * TL;
          }
        }

        double vL[b.ne];
        vL[0] =
            (-gk[ns - 1] + 0.5 * (pow(ul, 2) + pow(vl, 2) + pow(wl, 2))) / TL;
        vL[1] = -ul / TL;
        vL[2] = -vl / TL;
        vL[3] = -wl / TL;
        vL[4] = 1.0 / TL;
        for (int n = 0; n < ns - 1; n++) {
          vL[5 + n] = -(gk[n] - gk[ns - 1]) / TL;
        }

        double V[b.ne];
        for (int n = 0; n < b.ne; n++) {
          V[n] = 0.5 * (vR[n] + vL[n]);
        }
        double PHI = 0.5 * (phiR + phiL);

        double Fs = 0.0;
        for (int n = 0; n < ns; n++) {
          Fs += 0.5 * (rhoR * YR[n] + rhoL * YL[n]) * 0.5 * (skR[n] + skL[n]);
        }
        Fs *= U;
        double Ij = Fs + PHI - V[0] * b.iF(i, j, k, 0) -
                    V[1] * b.iF(i, j, k, 1) - V[2] * b.iF(i, j, k, 2) -
                    V[3] * b.iF(i, j, k, 3);

        // Then we need Species
        for (int n = 0; n < ns - 1; n++) {
          b.iF(i, j, k, 5 + n) =
              0.5 * (b.q(i, j, k, 5 + n) + b.q(i - 1, j, k, 5 + n)) * rho * U;
          Ij -= V[5 + n] * b.iF(i, j, k, 5 + n);
        }
        Ij /= V[4];
        Ij -= (Kj + Pj);

        b.iF(i, j, k, 4) = Ij + Kj + Pj;
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
        double Kj = rho * 0.5 *
                    (b.q(i, j, k, 1) * b.q(i, j - 1, k, 1) +
                     b.q(i, j, k, 2) * b.q(i, j - 1, k, 2) +
                     b.q(i, j, k, 3) * b.q(i, j - 1, k, 3)) *
                    V;

        double Pj =
            0.5 * (b.q(i, j - 1, k, 0) * (b.q(i, j, k, 1) * b.jsx(i, j, k) +
                                          b.q(i, j, k, 2) * b.jsy(i, j, k) +
                                          b.q(i, j, k, 3) * b.jsz(i, j, k)) +
                   b.q(i, j, k, 0) * (b.q(i, j - 1, k, 1) * b.jsx(i, j, k) +
                                      b.q(i, j - 1, k, 2) * b.jsy(i, j, k) +
                                      b.q(i, j - 1, k, 3) * b.jsz(i, j, k)));

        // solve for internal energy flux
        double gk[ns];
        double YR[ns];
        double YL[ns];
        // Compute nth species Y
        YR[ns - 1] = 1.0;
        YL[ns - 1] = 1.0;
        for (int n = 0; n < ns - 1; n++) {
          YR[n] = b.q(i, j, k, 5 + n);
          YL[n] = b.q(i, j - 1, k, 5 + n);
          YR[ns - 1] -= YR[n];
          YL[ns - 1] -= YL[n];
        }

        // right state
        double cvR = b.qh(i, j, k, 1) / b.qh(i, j, k, 0);
        double &TR = b.q(i, j, k, 4);
        double &rhoR = b.Q(i, j, k, 0);
        double RR = b.qh(i, j, k, 1) - cvR;
        double &ur = b.q(i, j, k, 1);
        double &vr = b.q(i, j, k, 2);
        double &wr = b.q(i, j, k, 3);
        double phiR =
            -RR * rhoR *
            (ur * b.jsx(i, j, k) + vr * b.jsy(i, j, k) + wr * b.jsz(i, j, k));

        double skR[ns];
        for (int n = 0; n < ns; n++) {
          if (YR[n] == 0.0) {
            skR[n] = 0.0;
            gk[n] = 0.0;
          } else {
            double cpk = th.cp0(n);
            double Rk = th.Ru / th.MW(n);
            double cvk = cpk - Rk;
            skR[n] = cvk * log(TR) - Rk * log(rhoR * YR[n]);
            double hk = b.qh(i, j, k, 5 + n);
            gk[n] = hk - skR[n] * TR;
          }
        }

        double vR[b.ne];
        vR[0] =
            (-gk[ns - 1] + 0.5 * (pow(ur, 2) + pow(vr, 2) + pow(wr, 2))) / TR;
        vR[1] = -ur / TR;
        vR[2] = -vr / TR;
        vR[3] = -wr / TR;
        vR[4] = 1.0 / TR;
        for (int n = 0; n < ns - 1; n++) {
          vR[5 + n] = -(gk[n] - gk[ns - 1]) / TR;
        }

        // left
        double cvL = b.qh(i, j - 1, k, 1) / b.qh(i, j - 1, k, 0);
        double &TL = b.q(i, j - 1, k, 4);
        double &rhoL = b.Q(i, j - 1, k, 0);
        double RL = b.qh(i, j - 1, k, 1) - cvL;
        double &ul = b.q(i, j - 1, k, 1);
        double &vl = b.q(i, j - 1, k, 2);
        double &wl = b.q(i, j - 1, k, 3);
        double phiL =
            -RL * rhoL *
            (ul * b.jsx(i, j, k) + vl * b.jsy(i, j, k) + wl * b.jsz(i, j, k));

        double skL[ns];
        for (int n = 0; n < ns; n++) {
          if (YL[n] == 0.0) {
            skL[n] = 0.0;
            gk[n] = 0.0;
          } else {
            double cpk = th.cp0(n);
            double Rk = th.Ru / th.MW(n);
            double cvk = cpk - Rk;
            skL[n] = cvk * log(TL) - Rk * log(rhoL * YL[n]);
            double hk = b.qh(i, j - 1, k, 5 + n);
            gk[n] = hk - skL[n] * TL;
          }
        }

        double vL[b.ne];
        vL[0] =
            (-gk[ns - 1] + 0.5 * (pow(ul, 2) + pow(vl, 2) + pow(wl, 2))) / TL;
        vL[1] = -ul / TL;
        vL[2] = -vl / TL;
        vL[3] = -wl / TL;
        vL[4] = 1.0 / TL;
        for (int n = 0; n < ns - 1; n++) {
          vL[5 + n] = -(gk[n] - gk[ns - 1]) / TL;
        }

        double Vj[b.ne];
        for (int n = 0; n < b.ne; n++) {
          Vj[n] = 0.5 * (vR[n] + vL[n]);
        }
        double PHI = 0.5 * (phiR + phiL);

        double Fs = 0.0;
        for (int n = 0; n < ns; n++) {
          Fs += 0.5 * (rhoR * YR[n] + rhoL * YL[n]) * 0.5 * (skR[n] + skL[n]);
        }
        Fs *= V;
        double Ij = Fs + PHI - Vj[0] * b.jF(i, j, k, 0) -
                    Vj[1] * b.jF(i, j, k, 1) - Vj[2] * b.jF(i, j, k, 2) -
                    Vj[3] * b.jF(i, j, k, 3);

        // Then we need Species
        for (int n = 0; n < ns - 1; n++) {
          b.jF(i, j, k, 5 + n) =
              0.5 * (b.q(i, j, k, 5 + n) + b.q(i, j - 1, k, 5 + n)) * rho * V;
          Ij -= Vj[5 + n] * b.jF(i, j, k, 5 + n);
        }
        Ij /= Vj[4];
        Ij -= (Kj + Pj);

        b.jF(i, j, k, 4) = Ij + Kj + Pj;
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
        double Kj = rho * 0.5 *
                    (b.q(i, j, k, 1) * b.q(i, j, k - 1, 1) +
                     b.q(i, j, k, 2) * b.q(i, j, k - 1, 2) +
                     b.q(i, j, k, 3) * b.q(i, j, k - 1, 3)) *
                    W;

        double Pj =
            0.5 * (b.q(i, j, k - 1, 0) * (b.q(i, j, k, 1) * b.ksx(i, j, k) +
                                          b.q(i, j, k, 2) * b.ksy(i, j, k) +
                                          b.q(i, j, k, 3) * b.ksz(i, j, k)) +
                   b.q(i, j, k, 0) * (b.q(i, j, k - 1, 1) * b.ksx(i, j, k) +
                                      b.q(i, j, k - 1, 2) * b.ksy(i, j, k) +
                                      b.q(i, j, k - 1, 3) * b.ksz(i, j, k)));
        // solve for internal energy flux
        double gk[ns];
        double YR[ns];
        double YL[ns];
        // Compute nth species Y
        YR[ns - 1] = 1.0;
        YL[ns - 1] = 1.0;
        for (int n = 0; n < ns - 1; n++) {
          YR[n] = b.q(i, j, k, 5 + n);
          YL[n] = b.q(i, j, k - 1, 5 + n);
          YR[ns - 1] -= YR[n];
          YL[ns - 1] -= YL[n];
        }

        // right state
        double cvR = b.qh(i, j, k, 1) / b.qh(i, j, k, 0);
        double &TR = b.q(i, j, k, 4);
        double &rhoR = b.Q(i, j, k, 0);
        double RR = b.qh(i, j, k, 1) - cvR;
        double &ur = b.q(i, j, k, 1);
        double &vr = b.q(i, j, k, 2);
        double &wr = b.q(i, j, k, 3);
        double phiR =
            -RR * rhoR *
            (ur * b.ksx(i, j, k) + vr * b.ksy(i, j, k) + wr * b.ksz(i, j, k));

        double skR[ns];
        for (int n = 0; n < ns; n++) {
          if (YR[n] == 0.0) {
            skR[n] = 0.0;
            gk[n] = 0.0;
          } else {
            double cpk = th.cp0(n);
            double Rk = th.Ru / th.MW(n);
            double cvk = cpk - Rk;
            skR[n] = cvk * log(TR) - Rk * log(rhoR * YR[n]);
            double hk = b.qh(i, j, k, 5 + n);
            gk[n] = hk - skR[n] * TR;
          }
        }

        double vR[b.ne];
        vR[0] =
            (-gk[ns - 1] + 0.5 * (pow(ur, 2) + pow(vr, 2) + pow(wr, 2))) / TR;
        vR[1] = -ur / TR;
        vR[2] = -vr / TR;
        vR[3] = -wr / TR;
        vR[4] = 1.0 / TR;
        for (int n = 0; n < ns - 1; n++) {
          vR[5 + n] = -(gk[n] - gk[ns - 1]) / TR;
        }

        // left
        double cvL = b.qh(i, j, k - 1, 1) / b.qh(i, j, k - 1, 0);
        double &TL = b.q(i, j, k - 1, 4);
        double &rhoL = b.Q(i, j, k - 1, 0);
        double RL = b.qh(i, j, k - 1, 1) - cvL;
        double &ul = b.q(i, j, k - 1, 1);
        double &vl = b.q(i, j, k - 1, 2);
        double &wl = b.q(i, j, k - 1, 3);
        double phiL =
            -RL * rhoL *
            (ul * b.ksx(i, j, k) + vl * b.ksy(i, j, k) + wl * b.ksz(i, j, k));

        double skL[ns];
        for (int n = 0; n < ns; n++) {
          if (YL[n] == 0.0) {
            skL[n] = 0.0;
            gk[n] = 0.0;
          } else {
            double cpk = th.cp0(n);
            double Rk = th.Ru / th.MW(n);
            double cvk = cpk - Rk;
            skL[n] = cvk * log(TL) - Rk * log(rhoL * YL[n]);
            double hk = b.qh(i, j, k - 1, 5 + n);
            gk[n] = hk - skL[n] * TL;
          }
        }

        double vL[b.ne];
        vL[0] =
            (-gk[ns - 1] + 0.5 * (pow(ul, 2) + pow(vl, 2) + pow(wl, 2))) / TL;
        vL[1] = -ul / TL;
        vL[2] = -vl / TL;
        vL[3] = -wl / TL;
        vL[4] = 1.0 / TL;
        for (int n = 0; n < ns - 1; n++) {
          vL[5 + n] = -(gk[n] - gk[ns - 1]) / TL;
        }

        double V[b.ne];
        for (int n = 0; n < b.ne; n++) {
          V[n] = 0.5 * (vR[n] + vL[n]);
        }
        double PHI = 0.5 * (phiR + phiL);

        double Fs = 0.0;
        for (int n = 0; n < ns; n++) {
          Fs += 0.5 * (rhoR * YR[n] + rhoL * YL[n]) * 0.5 * (skR[n] + skL[n]);
        }
        Fs *= W;
        double Ij = Fs + PHI - V[0] * b.kF(i, j, k, 0) -
                    V[1] * b.kF(i, j, k, 1) - V[2] * b.kF(i, j, k, 2) -
                    V[3] * b.kF(i, j, k, 3);

        // Then we need Species
        for (int n = 0; n < ns - 1; n++) {
          b.kF(i, j, k, 5 + n) =
              0.5 * (b.q(i, j, k, 5 + n) + b.q(i, j, k - 1, 5 + n)) * rho * W;
          Ij -= V[5 + n] * b.kF(i, j, k, 5 + n);
        }
        Ij /= V[4];
        Ij -= (Kj + Pj);

        b.kF(i, j, k, 4) = Ij + Kj + Pj;
      });
}

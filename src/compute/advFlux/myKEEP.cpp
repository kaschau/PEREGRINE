#include "block_.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"
#include <Kokkos_Core.hpp>

static void computeFlux(const block_ &b, fourDview &iF, const threeDview &isx,
                        const threeDview &isy, const threeDview &isz,
                        const int iMod, const int jMod, const int kMod) {

  // face flux range
  MDRange3 range(
      {b.ng, b.ng, b.ng},
      {b.ni + b.ng - 1 + iMod, b.nj + b.ng - 1 + jMod, b.nk + b.ng - 1 + kMod});

  Kokkos::parallel_for(
      "2nd order myKEEP face conv fluxes", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        // Compute face normal volume flux vector
        double uf =
            0.5 * (b.q(i, j, k, 1) + b.q(i - iMod, j - jMod, k - kMod, 1));
        double vf =
            0.5 * (b.q(i, j, k, 2) + b.q(i - iMod, j - jMod, k - kMod, 2));
        double wf =
            0.5 * (b.q(i, j, k, 3) + b.q(i - iMod, j - jMod, k - kMod, 3));

        double U = isx(i, j, k) * uf + isy(i, j, k) * vf + isz(i, j, k) * wf;

        double pf =
            0.5 * (b.q(i, j, k, 0) + b.q(i - iMod, j - jMod, k - kMod, 0));

        // Compute fluxes
        double rho;
        rho = 0.5 * (b.Q(i, j, k, 0) + b.Q(i - iMod, j - jMod, k - kMod, 0));

        // Continuity rho*Ui
        double Cj = rho * U;
        iF(i, j, k, 0) = Cj;

        // x momentum rho*u*Ui+ p*Ax
        iF(i, j, k, 1) = Cj * uf + pf * isx(i, j, k);

        // y momentum rho*v*Ui+ p*Ay
        iF(i, j, k, 2) = Cj * vf + pf * isy(i, j, k);

        // w momentum rho*w*Ui+ p*Az
        iF(i, j, k, 3) = Cj * wf + pf * isz(i, j, k);

        // Total energy (rhoE+ p)*Ui)
        double Kj = rho * 0.5 *
                    (b.q(i, j, k, 1) * b.q(i - iMod, j - jMod, k - kMod, 1) +
                     b.q(i, j, k, 2) * b.q(i - iMod, j - jMod, k - kMod, 2) +
                     b.q(i, j, k, 3) * b.q(i - iMod, j - jMod, k - kMod, 3)) *
                    U;

        double Pj =
            0.5 * (b.q(i - iMod, j - jMod, k - kMod, 0) *
                       (b.q(i, j, k, 1) * isx(i, j, k) +
                        b.q(i, j, k, 2) * isy(i, j, k) +
                        b.q(i, j, k, 3) * isz(i, j, k)) +
                   b.q(i, j, k, 0) *
                       (b.q(i - iMod, j - jMod, k - kMod, 1) * isx(i, j, k) +
                        b.q(i - iMod, j - jMod, k - kMod, 2) * isy(i, j, k) +
                        b.q(i - iMod, j - jMod, k - kMod, 3) * isz(i, j, k)));

        // solve for internal energy flux
        double cvR = b.qh(i, j, k, 1) / b.qh(i, j, k, 0);
        double &TR = b.q(i, j, k, 4);
        double &rhoR = b.Q(i, j, k, 0);
        double RR = b.qh(i, j, k, 1) - cvR;
        double hR = b.qh(i, j, k, 2) / rhoR;
        double eR = b.qh(i, j, k, 4);
        double &uR = b.q(i, j, k, 1);
        double &vR = b.q(i, j, k, 2);
        double &wR = b.q(i, j, k, 3);
        double sR = cvR * log(TR) - RR * log(rhoR);
        double phiR =
            -RR * rhoR *
            (uR * isx(i, j, k) + vR * isy(i, j, k) + wR * isz(i, j, k));

        double v0R =
            sR + (-hR + 0.5 * (pow(uR, 2) + pow(vR, 2) + pow(wR, 2))) / TR;
        double v1R = -uR / TR;
        double v2R = -vR / TR;
        double v3R = -wR / TR;
        double v4R = 1.0 / TR;

        // left
        double cvL = b.qh(i - iMod, j - jMod, k - kMod, 1) /
                     b.qh(i - iMod, j - jMod, k - kMod, 0);
        double &TL = b.q(i - iMod, j - jMod, k - kMod, 4);
        double &rhoL = b.Q(i - iMod, j - jMod, k - kMod, 0);
        double RL = b.qh(i - iMod, j - jMod, k - kMod, 1) - cvL;
        double hL = b.qh(i - iMod, j - jMod, k - kMod, 2) / rhoL;
        double eL = b.qh(i - iMod, j - jMod, k - kMod, 4);
        double &uL = b.q(i - iMod, j - jMod, k - kMod, 1);
        double &vL = b.q(i - iMod, j - jMod, k - kMod, 2);
        double &wL = b.q(i - iMod, j - jMod, k - kMod, 3);
        double sL = cvL * log(TL) - RL * log(rhoL);
        double phiL =
            -RL * rhoL *
            (uL * isx(i, j, k) + vL * isy(i, j, k) + wL * isz(i, j, k));

        double v0L =
            sL + (-hL + 0.5 * (pow(uL, 2) + pow(vL, 2) + pow(wL, 2))) / TL;
        double v1L = -uL / TL;
        double v2L = -vL / TL;
        double v3L = -wL / TL;
        double v4L = 1.0 / TL;

        double V0 = 0.5 * (v0R + v0L);
        double V1 = 0.5 * (v1R + v1L);
        double V2 = 0.5 * (v2R + v2L);
        double V3 = 0.5 * (v3R + v3L);
        double V4 = 0.5 * (v4R + v4L);
        double PHI = 0.5 * (phiR + phiL);

        // This comes from Eq. 4.5c of Tamdor, but there are consequences to the
        // form of G_v+1/2 (Fs_m+1/2)
        // If we use the cubic form
        //
        double Fs = rho * U * 0.5 * (sR + sL);
        //
        // Then we actually match the KEEPep scheme almost float for float,
        // so close that there has to be an equality between them.
        //
        // Whereas if we use either the quadratic or divergent forms
        // double Fs = 0.5 * (rhoR * sR + rhoL * sL) * U;
        // double Fs = 0.5 * (rhoR * uR * sR + rhoL * uL * sL) * iS(i, j, k);
        //
        // Then we are substantially different from the KEEPep scheme. In fact
        // the original KEEP scheme, none of the forms of Fs results in the same
        // answer. However, when we use the quadratic and divergent forms
        // We match exactly with v.\delF/delx evolution, where as for the cubic
        // form of Fs, we do not match v.\delF/\delx. wtf.

        double Ij = (Fs + PHI - V0 * iF(i, j, k, 0) - V1 * iF(i, j, k, 1) -
                     V2 * iF(i, j, k, 2) - V3 * iF(i, j, k, 3)) /
                        V4 -
                    Pj - Kj;
        Ij = (eL * TR + eR * TL) / (TL + TR) * U;

        iF(i, j, k, 4) = Ij + Kj + Pj;

        // Species
        for (int n = 0; n < b.ne - 5; n++) {
          iF(i, j, k, 5 + n) =
              0.5 *
              (b.Q(i, j, k, 5 + n) + b.Q(i - iMod, j - jMod, k - kMod, 5 + n)) *
              U;
        }
      });
}

void myKEEP(block_ &b) {
  computeFlux(b, b.iF, b.isx, b.isy, b.isz, 1, 0, 0);
  computeFlux(b, b.jF, b.jsx, b.jsy, b.jsz, 0, 1, 0);
  computeFlux(b, b.kF, b.ksx, b.ksy, b.ksz, 0, 0, 1);
}

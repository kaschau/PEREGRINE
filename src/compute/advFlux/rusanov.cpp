#include "block_.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"
#include <Kokkos_Core.hpp>

static void computeFlux(const block_ &b, fourDview &iF, const threeDview &iS,
                        const threeDview &isx, const threeDview &isy,
                        const threeDview &isz, const threeDview &inx,
                        const threeDview &iny, const threeDview &inz,
                        const int iMod, const int jMod, const int kMod) {

  // face flux range
  MDRange3 range(
      {b.ng, b.ng, b.ng},
      {b.ni + b.ng - 1 + iMod, b.nj + b.ng - 1 + jMod, b.nk + b.ng - 1 + kMod});
  Kokkos::parallel_for(
      "rusanov face conv fluxes", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double UR;
        double UL;

        double &ufR = b.q(i, j, k, 1);
        double &vfR = b.q(i, j, k, 2);
        double &wfR = b.q(i, j, k, 3);

        double &ufL = b.q(i - iMod, j - jMod, k - kMod, 1);
        double &vfL = b.q(i - iMod, j - jMod, k - kMod, 2);
        double &wfL = b.q(i - iMod, j - jMod, k - kMod, 3);

        UR = inx(i, j, k) * ufR + iny(i, j, k) * vfR + inz(i, j, k) * wfR;
        UL = inx(i, j, k) * ufL + iny(i, j, k) * vfL + inz(i, j, k) * wfL;

        double &rhoR = b.Q(i, j, k, 0);
        double &rhoL = b.Q(i - iMod, j - jMod, k - kMod, 0);

        double &pR = b.q(i, j, k, 0);
        double &pL = b.q(i - iMod, j - jMod, k - kMod, 0);

        double &ER = b.Q(i, j, k, 4);
        double &EL = b.Q(i - iMod, j - jMod, k - kMod, 4);

        // wave speed estimate
        double lam = fmax(abs(UL) + b.qh(i, j, k, 3),
                          abs(UR) + b.qh(i - iMod, j - jMod, k - kMod, 3)) *
                     iS(i, j, k);
        UR *= iS(i, j, k);
        UL *= iS(i, j, k);

        // Continuity rho*Ui
        double FrhoR, FrhoL;
        FrhoR = UR * rhoR;
        FrhoL = UL * rhoL;
        iF(i, j, k, 0) = 0.5 * (FrhoR + FrhoL - lam * (rhoR - rhoL));

        double FUR, FUL;
        // x momentum rho*u*Ui+ p*Ax
        FUR = UR * ufR * rhoR + pR * isx(i, j, k);
        FUL = UL * ufL * rhoL + pL * isx(i, j, k);
        iF(i, j, k, 1) = 0.5 * (FUR + FUL - lam * (rhoR * ufR - rhoL * ufL));

        // y momentum rho*v*Ui+ p*Ay
        FUR = UR * vfR * rhoR + pR * isy(i, j, k);
        FUL = UL * vfL * rhoL + pL * isy(i, j, k);
        iF(i, j, k, 2) = 0.5 * (FUR + FUL - lam * (rhoR * vfR - rhoL * vfL));

        // w momentum rho*w*Ui+ p*Az
        FUR = UR * wfR * rhoR + pR * isz(i, j, k);
        FUL = UL * wfL * rhoL + pL * isz(i, j, k);
        iF(i, j, k, 3) = 0.5 * (FUR + FUL - lam * (rhoR * wfR - rhoL * wfL));

        // Total energy (rhoE+ p)*Ui)
        double FER, FEL;
        FER = UR * (ER + pR);
        FEL = UL * (EL + pL);
        iF(i, j, k, 4) = 0.5 * (FER + FEL - lam * (ER - EL));

        // Species
        double FYiR, FYiL;
        double YiR, YiL;
        for (int n = 0; n < b.ne - 5; n++) {
          FYiR = b.Q(i, j, k, 5 + n) * UR;
          FYiL = b.Q(i - iMod, j - jMod, k - kMod, 5 + n) * UL;
          YiR = b.Q(i, j, k, 5 + n);
          YiL = b.Q(i - iMod, j - jMod, k - kMod, 5 + n);
          iF(i, j, k, 5 + n) = 0.5 * (FYiR + FYiL - lam * (YiR - YiL));
        }
      });
}

void rusanov(block_ &b) {
  computeFlux(b, b.iF, b.iS, b.isx, b.isy, b.isz, b.inx, b.iny, b.inz, 1, 0, 0);
  computeFlux(b, b.jF, b.jS, b.jsx, b.jsy, b.jsz, b.jnx, b.jny, b.jnz, 0, 1, 0);
  computeFlux(b, b.kF, b.kS, b.ksx, b.ksy, b.ksz, b.knx, b.kny, b.knz, 0, 0, 1);
}

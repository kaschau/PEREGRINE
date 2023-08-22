#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkosTypes.hpp"
#include "math.h"
#include "thtrdat_.hpp"

void KEEPdissipation(block_ &b) {

  const double kappa2 = 0.5;

  //-------------------------------------------------------------------------------------------|
  // i flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_i({b.ng, b.ng, b.ng},
                   {b.ni + b.ng, b.nj + b.ng - 1, b.nk + b.ng - 1});
  Kokkos::parallel_for(
      "Scalar Dissipation i face conv fluxes", range_i,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        const double eps2 =
            kappa2 * fmax(b.phi(i, j, k, 0), b.phi(i - 1, j, k, 0));

        // Compute face normal volume flux vector
        const double uf = 0.5 * (b.q(i, j, k, 1) + b.q(i - 1, j, k, 1));
        const double vf = 0.5 * (b.q(i, j, k, 2) + b.q(i - 1, j, k, 2));
        const double wf = 0.5 * (b.q(i, j, k, 3) + b.q(i - 1, j, k, 3));

        const double U =
            b.inx(i, j, k) * uf + b.iny(i, j, k) * vf + b.inz(i, j, k) * wf;

        const double a =
            (abs(U) + 0.5 * (b.qh(i, j, k, 3) + b.qh(i - 1, j, k, 3))) *
            b.iS(i, j, k);

        double rho2;
        rho2 = b.Q(i, j, k, 0) - b.Q(i - 1, j, k, 0);

        // Continuity dissipation
        double Dc = a * eps2 * rho2;
        b.iF(i, j, k, 0) = Dc;

        // u momentum dissipation
        double u2;
        u2 = b.Q(i, j, k, 1) - b.Q(i - 1, j, k, 1);
        double Dmu = a * eps2 * u2;
        b.iF(i, j, k, 1) = Dmu;

        // v momentum dissipation
        double v2;
        v2 = b.Q(i, j, k, 2) - b.Q(i - 1, j, k, 2);
        double Dmv = a * eps2 * v2;
        b.iF(i, j, k, 2) = Dmv;

        // w momentum dissipation
        double w2;
        w2 = b.Q(i, j, k, 3) - b.Q(i - 1, j, k, 3);
        double Dmw = a * eps2 * w2;
        b.iF(i, j, k, 3) = Dmw;

        // ke dissipation
        double Dk = -0.5 *
                        (pow(b.q(i, j, k, 1), 2.0) + pow(b.q(i, j, k, 2), 2.0) +
                         pow(b.q(i, j, k, 3), 2.0)) *
                        Dc +
                    b.q(i, j, k, 1) * Dmu + b.q(i, j, k, 2) * Dmv +
                    b.q(i, j, k, 3) * Dmw;
        // double k2 = 0.5 *
        //                 (pow(b.Q(i, j, k, 1), 2.0) + pow(b.Q(i,
        //                 j, k, 1), 2.0) +
        //                  pow(b.Q(i, j, k, 1), 2.0)) /
        //                 b.Q(i, j, k, 0) -
        //             0.5 * (pow(b.Q(i - 1, j, k, 1), 2.0) +
        //                    pow(b.Q(i - 1, j, k, 1), 2.0) +
        //                    pow(b.Q(i - 1, j, k, 1), 2.0) / b.Q(i
        //                    - 1, j, k, 0));

        // double Dk = a * eps2 * k2;

        // P dissipation
        double Dp = a * eps2 * (b.q(i, j, k, 0) - b.q(i - 1, j, k, 0));

        // solve for internal energy dissipation
        double cvR = b.qh(i, j, k, 1) / b.qh(i, j, k, 0);
        double &TR = b.q(i, j, k, 4);
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

        double V0 = 0.5 * (v0R + v0L);
        double V1 = 0.5 * (v1R + v1L);
        double V2 = 0.5 * (v2R + v2L);
        double V3 = 0.5 * (v3R + v3L);
        double V4 = 0.5 * (v4R + v4L);

        double rhos2 = rhoR * sR - rhoL * sL;
        double Ds = a * eps2 * rhos2;
        double De = (Ds - V0 * b.iF(i, j, k, 0) - V1 * b.iF(i, j, k, 1) -
                     V2 * b.iF(i, j, k, 2) - V3 * b.iF(i, j, k, 3)) /
                        V4 -
                    Dk;
        b.iF(i, j, k, 4) = De + Dk; // + Dp;

        // TODO Species???
        for (int n = 0; n < b.ne - 5; n++) {
          b.iF(i, j, k, 5 + n) =
              Dc * 0.5 * b.Q(i, j, k, 5 + n) + b.Q(i - 1, j, k, 5 + n);
        }

        // double s2 = b.s(i, j, k) - b.s(i - 1, j, k);
        // b.siF(i, j, k) = a * eps2 * s2;
      });
}

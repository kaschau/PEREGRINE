#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>
#include <math.h>
#include <numeric>

// References
//
// Large eddy simulation of subsonic and supersonicchannel flow at moderate
// Reynolds number
//
// INTERNATIONAL JOURNAL FOR NUMERICAL METHODS IN
// FLUIDSInt.J.Numer.Meth.Fluids2000;32: 369 – 406 E. Lenormand,  P. Sagautb,
// and  L. Ta Phuoc

PG_ABI void pgSmagorinsky(int count, pgIn *Jinv_, pgIn *Q_, pgIn *grads_,
                          pgIn *qh_, pgOut *qt_, const pgDims *d) {
  for (int e = 0; e < count; e++) {
    auto Jinv = as3(Jinv_[e]);
    auto Q = as4(Q_[e]);
    auto grads = as5(grads_[e]);
    auto qh = as4(qh_[e]);
    auto qt = as4(qt_[e]);
    const int ni = d[e].ni, nj = d[e].nj, nk = d[e].nk;

    MDRange3 range_cc({ng - 1, ng - 1, ng - 1}, {ni + ng, nj + ng, nk + ng});

    Kokkos::parallel_for(
        "Smagorinsky subgrid", range_cc,
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
          const double Cs = 0.18;
          const double Prt = 0.4;
          const double Sct = 1.0;

          const double &dudx = grads(i, j, k, 1, 0);
          const double &dudy = grads(i, j, k, 1, 1);
          const double &dudz = grads(i, j, k, 1, 2);

          const double &dvdx = grads(i, j, k, 2, 0);
          const double &dvdy = grads(i, j, k, 2, 1);
          const double &dvdz = grads(i, j, k, 2, 2);

          const double &dwdx = grads(i, j, k, 3, 0);
          const double &dwdy = grads(i, j, k, 3, 1);
          const double &dwdz = grads(i, j, k, 3, 2);

          double S[3][3];
          S[0][0] = dudx;
          S[1][1] = dvdy;
          S[2][2] = dwdz;

          S[0][1] = 0.5 * (dudy + dvdx);
          S[1][0] = S[0][1];
          S[0][2] = 0.5 * (dudz + dwdx);
          S[2][0] = S[0][2];
          S[1][2] = 0.5 * (dvdz + dwdy);
          S[2][1] = S[1][2];

          double magSij = 0.0;
          for (int l = 0; l < 3; l++) {
            for (int m = 0; m < 3; m++) {
              magSij += S[l][m] * S[l][m];
            }
          }
          magSij = sqrt(2.0 * magSij);

          double delta = cbrt(1.0 / Jinv(i, j, k));

          double nusgs = pow(Cs * delta, 2.0) * magSij;

          double musgs = nusgs * Q(i, j, k, 0);

          // Add sgs values to properties
          // viscocity
          qt(i, j, k, 0) += musgs;
          // thermal conductivity
          double kappasgs = musgs * qh(i, j, k, 1) / Prt;
          qt(i, j, k, 1) += kappasgs;
          // Diffusion coefficients mass
          for (int n = 0; n <= ne - 5; n++) {
            qt(i, j, k, 2 + n) += nusgs / Sct;
          }
        });
  }
}

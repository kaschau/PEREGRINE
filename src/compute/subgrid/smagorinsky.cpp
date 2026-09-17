#include "kernel.hpp"
#include <numeric>

// References
//
// Large eddy simulation of subsonic and supersonicchannel flow at moderate
// Reynolds number
//
// INTERNATIONAL JOURNAL FOR NUMERICAL METHODS IN
// FLUIDSInt.J.Numer.Meth.Fluids2000;32: 369 – 406 E. Lenormand,  P. Sagautb,
// and  L. Ta Phuoc

PG_RANGE(cellCenters)
struct smagorinsky {
  cellCenterIn Jinv, Q, grads, qh;
  cellCenterInOut qt;
  dims d;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const int ni = d->ni, nj = d->nj, nk = d->nk;

    const double Cs = 0.18;
    const double Prt = 0.4;
    const double Sct = 1.0;

    const double &dudx = grads(0, 0);
    const double &dudy = grads(0, 1);
    const double &dudz = grads(0, 2);

    const double &dvdx = grads(1, 0);
    const double &dvdy = grads(1, 1);
    const double &dvdz = grads(1, 2);

    const double &dwdx = grads(2, 0);
    const double &dwdy = grads(2, 1);
    const double &dwdz = grads(2, 2);

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

    double delta = cbrt(1.0 / Jinv());

    double nusgs = pow(Cs * delta, 2.0) * magSij;

    double musgs = nusgs * Q(0);

    // Add sgs values to properties
    // viscocity
    qt(0) += musgs;
    // thermal conductivity
    double kappasgs = musgs * qh(1) / Prt;
    qt(1) += kappasgs;
    // Diffusion coefficients mass
    for (int n = 0; n <= ne - 5; n++) {
      qt(2 + n) += nusgs / Sct;
    }
  }
};

PG_ABI void pgSmagorinsky(const smagorinsky &k, const pgTiling &t) {
  forCells("Smagorinsky subgrid", t, k);
}

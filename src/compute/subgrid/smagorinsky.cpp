#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkosTypes.hpp"
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

class NotImplemented : public std::logic_error {
public:
  NotImplemented() : std::logic_error("Function not yet implemented"){};
};

void smagorinsky(block_ &b) {
  NotImplemented();

  // MDRange3 range_cc({b.ng, b.ng, b.ng},
  //                   {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1});

  // Kokkos::parallel_for(
  //     "Smagorinsky subgrid", range_cc,
  //     KOKKOS_LAMBDA(const int i, const int j, const int k) {
  //       const double Cs = 0.18;
  //       const double Prt = 0.4;
  //       const double Sct = 1.0;

  //       double &dudx = b.dqdx(i, j, k, 1);
  //       double &dudy = b.dqdy(i, j, k, 1);
  //       double &dudz = b.dqdz(i, j, k, 1);

  //       double &dvdx = b.dqdx(i, j, k, 2);
  //       double &dvdy = b.dqdy(i, j, k, 2);
  //       double &dvdz = b.dqdz(i, j, k, 2);

  //       double &dwdx = b.dqdx(i, j, k, 3);
  //       double &dwdy = b.dqdy(i, j, k, 3);
  //       double &dwdz = b.dqdz(i, j, k, 3);

  //       double S[3][3];
  //       S[0][0] = dudx;
  //       S[1][1] = dvdy;
  //       S[2][2] = dwdz;

  //       S[0][1] = 0.5 * (dudy + dvdx);
  //       S[1][0] = S[0][1];
  //       S[0][2] = 0.5 * (dudz + dwdx);
  //       S[2][0] = S[0][2];
  //       S[1][2] = 0.5 * (dvdz + dwdy);
  //       S[2][1] = S[1][2];

  //       double magSij = 0.0;
  //       for (int l = 0; l < 3; l++) {
  //         for (int m = 0; m < 3; m++) {
  //           magSij += S[l][m] * S[l][m];
  //         }
  //       }
  //       magSij = sqrt(2.0 * magSij);

  //       double delta = pow(b.J(i, j, k), 1.0 / 3.0);

  //       double nusgs = pow(Cs * delta, 2.0) * magSij;

  //       double musgs = nusgs * b.Q(i, j, k, 0);

  //       // Add sgs values to properties
  //       // viscocity
  //       b.qt(i, j, k, 0) += musgs;
  //       // thermal conductivity
  //       double kappasgs = musgs * b.qh(i, j, k, 1) / Prt;
  //       b.qt(i, j, k, 1) += kappasgs;
  //       // Diffusion coefficients mass
  //       for (int n = 0; n <= b.ne - 5; n++) {
  //         b.qt(i, j, k, 2 + n) += nusgs / Sct;
  //       }
  //     });
}

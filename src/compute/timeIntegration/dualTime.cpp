#include "Kokkos_Core.hpp"
#include "array"
#include "block_.hpp"
#include "kokkosTypes.hpp"
#include "math.h"
#include "thtrdat_.hpp"
#include "vector"

//---------------------------------------------------------------------------------------------|
//
// Dual Time with preconditioning from
//
//     Preconditioning applied to variable and constant density flows
//     Weiss, Jonathan M. and Smith, Wayne A.
//     AIAA Journal
//     1995
//     doi: 10.2514/3.12946
//
//---------------------------------------------------------------------------------------------|

void dQdt(block_ &b, const double &dt) {
  //-------------------------------------------------------------------------------------------|
  // Add to dQ with real time derivative source term
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng, b.ng, b.ng, 0},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
  Kokkos::parallel_for(
      "dQdt", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        b.dQ(i, j, k, l) -= (3.0 * b.Q(i, j, k, l) - 4.0 * b.Qn(i, j, k, l) +
                             b.Qnm1(i, j, k, l)) /
                            (2 * dt);
      });
}

void localDtau(block_ &b, const bool &viscous) {
  //-------------------------------------------------------------------------------------------|
  // Compute local pseudo time step
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_cc({b.ng, b.ng, b.ng},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1});
  Kokkos::parallel_for(
      "localDtau", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        // Cell lengths
        double dI = sqrt(pow(b.ixc(i + 1, j, k) - b.ixc(i, j, k), 2.0) +
                         pow(b.iyc(i + 1, j, k) - b.iyc(i, j, k), 2.0) +
                         pow(b.izc(i + 1, j, k) - b.izc(i, j, k), 2.0));
        double dJ = sqrt(pow(b.jxc(i, j + 1, k) - b.jxc(i, j, k), 2.0) +
                         pow(b.jyc(i, j + 1, k) - b.jyc(i, j, k), 2.0) +
                         pow(b.jzc(i, j + 1, k) - b.jzc(i, j, k), 2.0));
        double dK = sqrt(pow(b.kxc(i, j, k + 1) - b.kxc(i, j, k), 2.0) +
                         pow(b.kyc(i, j, k + 1) - b.kyc(i, j, k), 2.0) +
                         pow(b.kzc(i, j, k + 1) - b.kzc(i, j, k), 2.0));

        // Find max convective CFL
        double &u = b.q(i, j, k, 1);
        double &v = b.q(i, j, k, 2);
        double &w = b.q(i, j, k, 3);

        double uI =
            sqrt(pow(0.5 * (b.inx(i, j, k) + b.inx(i + 1, j, k)) * u, 2.0) +
                 pow(0.5 * (b.iny(i, j, k) + b.iny(i + 1, j, k)) * v, 2.0) +
                 pow(0.5 * (b.inz(i, j, k) + b.inz(i + 1, j, k)) * w, 2.0));
        double uJ =
            sqrt(pow(0.5 * (b.jnx(i, j, k) + b.jnx(i, j + 1, k)) * u, 2.0) +
                 pow(0.5 * (b.jny(i, j, k) + b.jny(i, j + 1, k)) * v, 2.0) +
                 pow(0.5 * (b.jnz(i, j, k) + b.jnz(i, j + 1, k)) * w, 2.0));
        double uK =
            sqrt(pow(0.5 * (b.knx(i, j, k) + b.knx(i, j, k + 1)) * u, 2.0) +
                 pow(0.5 * (b.kny(i, j, k) + b.kny(i, j, k + 1)) * v, 2.0) +
                 pow(0.5 * (b.knz(i, j, k) + b.knz(i, j, k + 1)) * w, 2.0));

        double &c = b.qh(i, j, k, 3);

        double pseudoCFL = 0.5;
        double pseudoVNN = 0.1;

        double nu;
        if (viscous) {
          nu = b.qt(i, j, k, 0) / b.Q(i, j, k, 0);
        } else {
          nu = 0.0;
        }

        double dtau = 1.0e16;
        if (b.ni > 2) {
          dtau = fmin(dtau, pseudoCFL * dI / (uI + c));
          if (viscous) {
            dtau = fmin(dtau, pseudoVNN * pow(dI, 2.0) / nu);
          }
        }
        if (b.nj > 2) {
          dtau = fmin(dtau, pseudoCFL * dJ / (uJ + c));
          if (viscous) {
            dtau = fmin(dtau, pseudoVNN * pow(dJ, 2.0) / nu);
          }
        }
        if (b.nk > 2) {
          dtau = fmin(dtau, pseudoCFL * dK / (uK + c));
          if (viscous) {
            dtau = fmin(dtau, pseudoVNN * pow(dK, 2.0) / nu);
          }
        }

        b.dtau(i, j, k) = dtau;
      });
}

void DTrk3s1(block_ &b) {
  //-------------------------------------------------------------------------------------------|
  // Apply RK3 stage 1
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng, b.ng, b.ng, 0},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
  Kokkos::parallel_for(
      "DTrk3 stage 1", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        b.q(i, j, k, l) = b.Q0(i, j, k, l) + b.dtau(i, j, k) * b.dQ(i, j, k, l);
      });
}

void DTrk3s2(block_ &b) {
  //-------------------------------------------------------------------------------------------|
  // Apply RK3 stage 2
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng, b.ng, b.ng, 0},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
  Kokkos::parallel_for(
      "DTrk3 stage 2", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        b.q(i, j, k, l) = 0.75 * b.Q0(i, j, k, l) + 0.25 * b.q(i, j, k, l) +
                          0.25 * b.dQ(i, j, k, l) * b.dtau(i, j, k);
      });
}

void DTrk3s3(block_ &b) {
  //-------------------------------------------------------------------------------------------|
  // Apply RK3 stage 3
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng, b.ng, b.ng, 0},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
  Kokkos::parallel_for(
      "DTrk3 stage 3", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        b.q(i, j, k, l) = (b.Q0(i, j, k, l) + 2.0 * b.q(i, j, k, l) +
                           2.0 * b.dQ(i, j, k, l) * b.dtau(i, j, k)) /
                          3.0;
      });
}

std::array<std::vector<double>, 2> residual(std::vector<block_> &mb) {
  //-------------------------------------------------------------------------------------------|
  // Compute max residual for each primative
  //-------------------------------------------------------------------------------------------|
  int ne = mb[0].ne;
  double rPmax, rUmax, rVmax, rWmax, rTmax;
  double rPrms, rUrms, rVrms, rWrms, rTrms;
  std::array<std::vector<double>, 2> returnResid;

  for (int l = 0; l < ne; l++) {
    returnResid[0].push_back(0.0);
    returnResid[1].push_back(0.0);
  }

  // First we will find the max L_inf residual for each p,u,v,w,T primative
  // RECALL: Q0 array is actually primatives when using dual time
  for (block_ b : mb) {

    MDRange3 range_cc({b.ng, b.ng, b.ng},
                      {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1});
    Kokkos::parallel_reduce(
        "residual", range_cc,
        KOKKOS_LAMBDA(const int i, const int j, const int k, double &resPmax,
                      double &resUmax, double &resVmax, double &resWmax,
                      double &resTmax, double &resPrms, double &resUrms,
                      double &resVrms, double &resWrms, double &resTrms

        ) {
          double res;
          // Pressure
          res = abs(b.q(i, j, k, 0) - b.Q0(i, j, k, 0));
          resPmax = fmax(res, resPmax);
          resPrms += pow(res, 2.0);

          // u-velocity
          res = abs(b.q(i, j, k, 1) - b.Q0(i, j, k, 1));
          resUmax = fmax(res, resUmax);
          resUrms += pow(res, 2.0);

          // v-velocity
          res = abs(b.q(i, j, k, 2) - b.Q0(i, j, k, 2));
          resVmax = fmax(res, resVmax);
          resVrms += pow(res, 2.0);

          // w-velocity
          res = abs(b.q(i, j, k, 3) - b.Q0(i, j, k, 3));
          resWmax = fmax(res, resWmax);
          resWrms += pow(res, 2.0);

          // temperature
          res = abs(b.q(i, j, k, 4) - b.Q0(i, j, k, 4));
          resTmax = fmax(res, resTmax);
          resTrms += pow(res, 2.0);
        },
        Kokkos::Max<double>(rPmax), Kokkos::Max<double>(rUmax),
        Kokkos::Max<double>(rVmax), Kokkos::Max<double>(rWmax),
        Kokkos::Max<double>(rTmax), Kokkos::Sum<double>(rPrms),
        Kokkos::Sum<double>(rUrms), Kokkos::Sum<double>(rVrms),
        Kokkos::Sum<double>(rWrms), Kokkos::Sum<double>(rTrms));

    returnResid[0][0] = fmax(rPmax, returnResid[0][0]);
    returnResid[0][1] = fmax(rUmax, returnResid[0][1]);
    returnResid[0][2] = fmax(rVmax, returnResid[0][2]);
    returnResid[0][3] = fmax(rWmax, returnResid[0][3]);
    returnResid[0][4] = fmax(rTmax, returnResid[0][4]);

    returnResid[1][0] += rPrms;
    returnResid[1][1] += rUrms;
    returnResid[1][2] += rVrms;
    returnResid[1][3] += rWrms;
    returnResid[1][4] += rTrms;

    // Species
    for (int n = 5; n < ne; n++) {
      double rYmax;
      double rYrms;
      Kokkos::parallel_reduce(
          "residual", range_cc,
          KOKKOS_LAMBDA(const int i, const int j, const int k, double &resYmax,
                        double &resYrms) {
            double res;
            res = abs(b.q(i, j, k, n) - b.Q0(i, j, k, n));
            resYmax = fmax(res, resYmax);
            resYrms += pow(res, 2.0);
          },
          Kokkos::Max<double>(rYmax), Kokkos::Sum<double>(rYrms));
      returnResid[0][n] = fmax(rYmax, returnResid[0][n]);
      returnResid[1][n] += rYrms;
    }
  }
  return returnResid;
}

void invertDQ(block_ &b, const double &dt, const thtrdat_ &th,
              const bool &viscous) {
  //-------------------------------------------------------------------------------------------|
  // Solve (\Gamma + dqdQ) dq = dQ to solver for dqdt
  //
  // The premultiplying matrix takes the form of Weiss and Smith
  //
  // Preconditioning applied to variable and constant density flows
  // Weiss, Jonathan M. and Smith, Wayne A.
  // AIAA Journal
  // 1995
  // doi: 10.2514/3.12946
  //-------------------------------------------------------------------------------------------|

  MDRange3 range_cc({b.ng, b.ng, b.ng},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1});

#ifndef NSCOMPILE
  Kokkos::Experimental::UniqueToken<execSpace> token;
  int numIds = token.size();
  const int ne = b.ne;
  threeDview GdQ("GdQ", numIds, ne, ne);
  twoDview tempRow("tempRow", numIds, ne);
  twoDviewInt perm("perm", numIds, ne);
#endif

#ifdef NSCOMPILE
#define GdQ(INDEX, INDEX1) GdQ[INDEX][INDEX1]
#define perm(INDEX) perm[INDEX]
#define tempRow(INDEX) tempRow[INDEX]
#define ne 5 + NS - 1
#else
#define GdQ(INDEX, INDEX1) GdQ(id, INDEX, INDEX1)
#define perm(INDEX) perm(id, INDEX)
#define tempRow(INDEX) tempRow(id, INDEX)
#endif

#ifndef NSCOMPILE
  const int ns = th.ns;
  twoDview Y("Y", numIds, ns);
  twoDview rho_Y("rho_Y", numIds, ns);
#endif

#ifdef NSCOMPILE
#define Y(INDEX) Y[INDEX]
#define rho_Y(INDEX) rho_Y[INDEX]
#define ns NS
#else
#define Y(INDEX) Y(id, INDEX)
#define rho_Y(INDEX) rho_Y(id, INDEX)
#endif

  Kokkos::parallel_for(
      "dq = (Gamma + dqdQ)^{-1} dQ", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {

#ifdef NSCOMPILE
        double GdQ(ne, ne);
        int perm(ne);
        double tempRow(ne);
#else
        int id = token.acquire();
#endif

        ////////////////////////////////////////////////
        ///// COMPUTE GdQ MATRIX
        ///// Sum of Preconditioning matrix and
        ///// convservative to primative variable
        ///// transformation
        /////
        /////  \Gamma + 3*dtau / (2*dt) dQdq
        /////
        ////////////////////////////////////////////////
        double &p = b.q(i, j, k, 0);
        double &u = b.q(i, j, k, 1);
        double &v = b.q(i, j, k, 2);
        double &w = b.q(i, j, k, 3);
        double &T = b.q(i, j, k, 4);
        double &rho = b.Q(i, j, k, 0);
#ifdef NSCOMPILE
        double Y(ns);
        double rho_Y(ns);
#endif
        double cp = b.qh(i, j, k, 1);
        double H = b.qh(i, j, k, 2) / rho +
                   0.5 * (pow(u, 2.0) + pow(v, 2.0) + pow(w, 2.0));
        double c = b.qh(i, j, k, 3);
        // Compute nth species Y
        Y(ns - 1) = 1.0;
        double denom = 0.0;
        for (int n = 0; n < ns - 1; n++) {
          Y(n) = b.q(i, j, k, 5 + n);
          Y(ns - 1) -= Y(n);
          denom += Y(n) / th.MW(n);
        }
        denom += Y(ns - 1) / th.MW(ns - 1);

        // Compute MWmix
        double MWmix = 0.0;
        for (int n = 0; n <= ns - 1; n++) {
          double X = Y(n) / th.MW(n) / denom;
          MWmix += th.MW(n) * X;
        }

        // Compute required derivatives
        double rho_p = rho / p;
        double rho_T = -rho / T;

        for (int n = 0; n < ns - 1; n++) {
          rho_Y(n) = -rho * (MWmix * (1.0 / th.MW(n) - 1.0 / th.MW(ns - 1)));
        }

        /////////////////////////////////////////////////
        // The preconditioning and transformation matrix
        // share a very similar form, only differing by
        // the multiplier of the first column, and the
        // multiplication of the time derivatives for
        // the prim/cons transformation matrix.
        /////////////////////////////////////////////////
        for (int l = 0; l < ne; l++) {
          for (int m = 0; m < ne; m++) {
            GdQ(l, m) = 0.0;
          }
        }
        double Thetas[2];
        double mults[2];

        // Prematrix multipliers (constants)
        mults[0] = 1.0;
        mults[1] = 3.0 / 2.0 * b.dtau(i, j, k) / dt;

        // Reference velocity for preconditioning theta
        double U = abs(sqrt(pow(u, 2.0) + pow(v, 2.0) + pow(w, 2.0)));
        double eps = 1.0e-5;
        double Ur;

        // Ideal gas reference velocity
        if (U < eps * c) {
          Ur = eps * c;
        } else if (U > eps * c && U < c) {
          Ur = U;
        } else {
          Ur = c;
        }
        // // Incompressible reference velocity
        // double Umax = 5.0; // <- Case specific
        // if (U < epc) {
        //   Ur = eps * Umax;
        // } else {
        //   Ur = U;
        // }

        // Limit reference velocity by viscosity
        if (viscous) {
          double dI = sqrt(pow(b.ixc(i + 1, j, k) - b.ixc(i, j, k), 2.0) +
                           pow(b.iyc(i + 1, j, k) - b.iyc(i, j, k), 2.0) +
                           pow(b.izc(i + 1, j, k) - b.izc(i, j, k), 2.0));
          double dJ = sqrt(pow(b.jxc(i, j + 1, k) - b.jxc(i, j, k), 2.0) +
                           pow(b.jyc(i, j + 1, k) - b.jyc(i, j, k), 2.0) +
                           pow(b.jzc(i, j + 1, k) - b.jzc(i, j, k), 2.0));
          double dK = sqrt(pow(b.kxc(i, j, k + 1) - b.kxc(i, j, k), 2.0) +
                           pow(b.kyc(i, j, k + 1) - b.kyc(i, j, k), 2.0) +
                           pow(b.kzc(i, j, k + 1) - b.kzc(i, j, k), 2.0));
          double nu = b.qt(i, j, k, 0) / b.Q(i, j, k, 0);
          if (b.ni > 2) {
            Ur = fmin(Ur, dI / nu);
          }
          if (b.nj > 2) {
            Ur = fmin(Ur, dJ / nu);
          }
          if (b.nk > 2) {
            Ur = fmin(Ur, dK / nu);
          }
        }

        // Thetas (just rho_p for dQdq)
        Thetas[0] = 1.0 / pow(Ur, 2.0) - rho_T / (rho * cp);
        Thetas[1] = rho_p;

        ///////////////////////////////////////////////////////////////////
        // Gamma and dQdq are constricted in the following blocks
        // |-----------------------|------------------|
        // |                       |                  |
        // |         (1)           |       (2)        |
        // |      Single Comp      |  Prims/Species   |
        // |       Primatives      |                  |
        // |                       |                  |
        // |-----------------------|------------------|
        // |                       |                  |
        // |         (3)           |       (4)        |
        // |     Species/Prims     | Species/Species  |
        // |                       |                  |
        // |                       |                  |
        // |-----------------------|------------------|
        //
        // In a column by column manner
        ///////////////////////////////////////////////////////////////////

        for (int p = 0; p < 2; p++) {
          double Theta = Thetas[p];
          double mult = mults[p];

          // Block (1)
          // First column
          GdQ(0, 0) += mult * Theta;
          GdQ(1, 0) += mult * Theta * u;
          GdQ(2, 0) += mult * Theta * v;
          GdQ(3, 0) += mult * Theta * w;
          GdQ(4, 0) += mult * (Theta * H + T * rho_T / rho);

          // Second column
          GdQ(0, 1) += mult * 0.0;
          GdQ(1, 1) += mult * rho;
          GdQ(2, 1) += mult * 0.0;
          GdQ(3, 1) += mult * 0.0;
          GdQ(4, 1) += mult * rho * u;

          // Third column
          GdQ(0, 2) += mult * 0.0;
          GdQ(1, 2) += mult * 0.0;
          GdQ(2, 2) += mult * rho;
          GdQ(3, 2) += mult * 0.0;
          GdQ(4, 2) += mult * rho * v;

          // Fourth column
          GdQ(0, 3) += mult * 0.0;
          GdQ(1, 3) += mult * 0.0;
          GdQ(2, 3) += mult * 0.0;
          GdQ(3, 3) += mult * rho;
          GdQ(4, 3) += mult * rho * w;

          // Fifth column
          GdQ(0, 4) += mult * rho_T;
          GdQ(1, 4) += mult * rho_T * u;
          GdQ(2, 4) += mult * rho_T * v;
          GdQ(3, 4) += mult * rho_T * w;
          GdQ(4, 4) += mult * (rho_T * H + rho * cp);

          for (int n = 5; n < ne; n++) {
            // Block (2) nth column
            GdQ(0, n) += mult * rho_Y(n - 5);
            GdQ(1, n) += mult * rho_Y(n - 5) * u;
            GdQ(2, n) += mult * rho_Y(n - 5) * v;
            GdQ(3, n) += mult * rho_Y(n - 5) * w;
            double h_y = b.qh(i, j, k, n) - b.qh(i, j, k, ne);
            GdQ(4, n) += mult * (H * rho_Y(n - 5) + rho * h_y);
            // Block (3)
            GdQ(n, 0) += mult * Theta * Y(n - 5);
            GdQ(n, 1) += mult * 0.0;
            GdQ(n, 2) += mult * 0.0;
            GdQ(n, 3) += mult * 0.0;
            GdQ(n, 4) += mult * rho_T * Y(n - 5);
          }

          // Block (4)
          for (int n = 5; n < ne; n++) {
            for (int q = 5; q < ne; q++) {
              GdQ(q, n) += mult * Y(q - 5) * rho_Y(n - 5);
            }
          }
          for (int n = 5; n < ne; n++) {
            GdQ(n, n) += mult * rho;
          }
        }

        /////////////////////////////////////////////////////////////////////////////
        // Perform LU decomposition with partial pivoting
        // Routine modifies GdQ in place resulting in a
        // strictly lower triangle matrix with 1.0 along the diagonal
        // and an upper triangular matrix including the diagonal.
        /////////////////////////////////////////////////////////////////////////////

        for (int l = 0; l < ne; l++) {
          perm(l) = l;
        }

        for (int l = 0; l < ne; l++) {
          int pivotInd = 0;
          double pivot = 0.0;
          int tempInd;
          for (int m = l; m < ne; m++)
            if (abs(GdQ(m, l)) > abs(pivot)) {
              pivot = GdQ(m, l);
              pivotInd = m;
            }

          for (int p = 0; p < ne; p++) {
            tempRow(p) = GdQ(l, p);
            GdQ(l, p) = GdQ(pivotInd, p);
            GdQ(pivotInd, p) = tempRow(p);
          }

          tempInd = perm(l);
          perm(l) = perm(pivotInd);
          perm(pivotInd) = tempInd;

          for (int p = l + 1; p < ne; p++) {
            double temp;
            temp = GdQ(p, l) /= GdQ(l, l);
            for (int q = l + 1; q < ne; q++) {
              GdQ(p, q) -= temp * GdQ(l, q);
            }
          }
        }

        // Row permute dQ to match LU
        for (int l = 0; l < ne; l++) {
          tempRow(l) = b.dQ(i, j, k, perm(l));
        }
        for (int l = 0; l < ne; l++) {
          b.dQ(i, j, k, l) = tempRow(l);
        }

        // Solve Ax = b where A = LU by first solving for
        //
        // Lz = a then Ux=z
        //
        // Form of the equations is actually
        //
        // LU(dq) = dQ
        //
        // So begin with Lz = dQ where tempRow = z

        for (int l = 0; l < ne; l++) {
          for (int q = 0; q < l; q++) {
            tempRow(l) -= GdQ(l, q) * tempRow(q);
          }
        }

        // Now solve Ux=z which is actually
        //
        // U(dq) = tempRow
        //
        // Recall we are working with primatives so we will modify the dQ view
        // in place with the resultant dq values (as x)

        for (int l = ne - 1; l > -1; l--) {
          b.dQ(i, j, k, l) = tempRow(l);
          for (int q = ne - 1; q > l; q--) {
            b.dQ(i, j, k, l) -= GdQ(l, q) * b.dQ(i, j, k, q);
          }
          b.dQ(i, j, k, l) /= GdQ(l, l);
        }
      });
}

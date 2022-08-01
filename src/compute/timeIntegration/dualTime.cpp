#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkos_types.hpp"
#include "math.h"
#include "thtrdat_.hpp"
#include "vector"

void dQdt(block_ b, const double dt) {
  //-------------------------------------------------------------------------------------------|
  // Start off dQ with real time derivative source term
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng, b.ng, b.ng, 0},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
  Kokkos::parallel_for(
      "dQdt", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        b.dQ(i, j, k, l) = (3.0 * b.Q(i, j, k, l) - 4.0 * b.Qn(i, j, k, l) +
                            b.Qnm1(i, j, k, l)) /
                           (2 * dt);
      });
}

void DTrk2s1(block_ b, const double dt) {
  //-------------------------------------------------------------------------------------------|
  // Apply RK2 stage 1
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng, b.ng, b.ng, 0},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
  Kokkos::parallel_for(
      "Dual time rk2 stage 1", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        b.q(i, j, k, l) += b.dQ(i, j, k, l) * dt;
      });
}

void DTrk2s2(block_ b, const double dt) {
  //-------------------------------------------------------------------------------------------|
  // Apply RK2 stage 2
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng, b.ng, b.ng, 0},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
  Kokkos::parallel_for(
      "Dual TIme rk2 stage 2", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        b.q(i, j, k, l) = 0.5 * b.Q0(i, j, k, l) +
                          0.5 * (b.q(i, j, k, l) + dt * b.dQ(i, j, k, l));
      });
}

std::vector<double> residual(std::vector<block_> mb) {
  //-------------------------------------------------------------------------------------------|
  // Compute max residual for each primative
  //-------------------------------------------------------------------------------------------|

  int ne = mb[0].ne;
  double rP, rU, rV, rW, rT, rYi;
  std::vector<double> returnResid;
  for (int l = 0; l < ne; l++) {
    returnResid.push_back(0.0);
  }

  // First we will find the max L_inf residual for each p,u,v,w,T primative
  // RECALL: Q0 array is actually primatives when using dual time
  for (block_ b : mb) {

    MDRange3 range_cc({b.ng, b.ng, b.ng},
                      {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1});
    Kokkos::parallel_reduce(
        "residual", range_cc,
        KOKKOS_LAMBDA(const int i, const int j, const int k, double &rP,
                      double &rU, double &rV, double &rW, double &rT) {
          rP = abs(b.q(i, j, k, 0) - b.Q0(i, j, k, 0));
          rU = abs(b.q(i, j, k, 1) - b.Q0(i, j, k, 1));
          rV = abs(b.q(i, j, k, 2) - b.Q0(i, j, k, 2));
          rW = abs(b.q(i, j, k, 3) - b.Q0(i, j, k, 3));
          rT = abs(b.q(i, j, k, 4) - b.Q0(i, j, k, 4));
        },
        Kokkos::Max<double>(rP), Kokkos::Max<double>(rU),
        Kokkos::Max<double>(rV), Kokkos::Max<double>(rW),
        Kokkos::Max<double>(rT));

    returnResid[0] = fmax(rP, returnResid[0]);
    returnResid[1] = fmax(rU, returnResid[1]);
    returnResid[2] = fmax(rV, returnResid[2]);
    returnResid[3] = fmax(rW, returnResid[3]);
    returnResid[4] = fmax(rT, returnResid[4]);

    for (int n = 5; n < ne; n++) {
      Kokkos::parallel_reduce(
          "residual", range_cc,
          KOKKOS_LAMBDA(const int i, const int j, const int k, double &rYi) {
            rYi = abs(b.q(i, j, k, n) - b.Q0(i, j, k, n));
          },
          Kokkos::Max<double>(rYi));
      returnResid[n] = fmax(rP, returnResid[n]);
    }
  }

  return returnResid;
}

void invertDQ(block_ b, const double dt, const double dtau, const thtrdat_ th) {
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
  Kokkos::Experimental::UniqueToken<exec_space> token;
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
#define rho_Y(INDEX) Y[INDEX]
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
        double Y(ne);
        double rho_Y(ne);
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
        double H = b.qh(i, j, k, 2);
        double cp = b.qh(i, j, k, 1);
        // Compute nth species Y
        Y(ns - 1) = 1.0;
        double denom = 0.0;
        for (int n = 0; n < ns - 1; n++) {
          Y(n) = b.q(i, j, k, 5 + n);
          Y(ns - 1) -= Y(n);
          denom += Y(n) / th.MW(n);
        }
        denom += Y(ns) / th.MW(ns);

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
          rho_Y(n) = -rho * (MWmix * (1.0 / th.MW(n) - 1.0 / th.MW(ns)));
        }

        ////////////////////////////////////
        // The preconditioning and transformation matrix
        // share a very similar form, only differing by
        // the multiplier of the first column, and the
        // multiplication of the time derivatives for
        // the prim/cons transformation matrix.
        ////////////////////////////////////
        for (int l = 0; l < ne; l++) {
          for (int m = 0; m < ne; m++) {
            GdQ(l, m) = 0.0;
          }
        }
        double phis[2];
        double mults[2];

        mults[0] = 1.0;
        mults[1] = 3.0 / 2.0 * dtau / dt;

        phis[0] = 1.0;
        phis[1] = rho_p;

        for (int p = 0; p < 2; p++) {
          double phi = phis[p];
          double mult = mults[p];

          // First column
          GdQ(0, 0) += mult * phi;
          GdQ(1, 0) += mult * phi * u;
          GdQ(2, 0) += mult * phi * v;
          GdQ(3, 0) += mult * phi * w;
          GdQ(4, 0) += mult * phi * H +
                       T * rho_T / rho; // <- For an ideal gas this is phi*H - 1

          GdQ(0, 1) += mult * 0.0;
          GdQ(1, 1) += mult * rho;
          GdQ(2, 1) += mult * 0.0;
          GdQ(3, 1) += mult * 0.0;
          GdQ(4, 1) += mult * rho * u;

          GdQ(0, 2) += mult * 0.0;
          GdQ(1, 2) += mult * 0.0;
          GdQ(2, 2) += mult * rho;
          GdQ(3, 2) += mult * 0.0;
          GdQ(4, 2) += mult * rho * v;

          GdQ(0, 3) += mult * 0.0;
          GdQ(1, 3) += mult * 0.0;
          GdQ(2, 3) += mult * 0.0;
          GdQ(3, 3) += mult * rho;
          GdQ(4, 3) += mult * rho * w;

          GdQ(0, 4) += mult * rho_T;
          GdQ(1, 4) += mult * rho_T * u;
          GdQ(2, 4) += mult * rho_T * v;
          GdQ(3, 4) += mult * rho_T * w;
          GdQ(4, 4) += mult * rho_T * H + rho * cp;

          for (int n = 5; n < ne; n++) {
            GdQ(0, n) += mult * rho_Y(n);
            GdQ(1, n) += mult * rho_Y(n) * u;
            GdQ(2, n) += mult * rho_Y(n) * v;
            GdQ(3, n) += mult * rho_Y(n) * w;
            GdQ(4, n) += mult * rho_T * Y(n);
            GdQ(n, 0) += mult * phi * Y(n);
            GdQ(n, 1) += mult * 0.0;
            GdQ(n, 2) += mult * 0.0;
            GdQ(n, 3) += mult * 0.0;
            GdQ(n, 4) += mult * rho_T * Y(n);
          }
          for (int n = 5; n < ne; n++) {
            for (int p = 5; p < ne; p++) {
              GdQ(p, n) += mult * Y(p) * rho_Y(n);
            }
          }
          for (int n = 5; n < ne; n++) {
            GdQ(n, n) += mult * rho;
          }
        }

        ///////////////////////////////////////////////////////////////////////////
        printf("Gamma:\n");
        for (int q = 0; q < ne; q++) {
          for (int l = 0; l < ne; l++) {
            printf("%lf ", GdQ(q, l));
          }
          printf("\n");
        }
        ///////////////////////////////////////////////////////////////////////////

        // Perform LU decomposition with partial pivoting
        // Routine modified GdQ in place resulting in a
        // strictly lower triangle matrix with 1.0 along the diagonal
        // and an upper triangular matrix including the diagonal.

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

        /////////////////////////////////////////////////////////////////////////////
        printf("LU Decomposition:\n");
        for (int q = 0; q < ne; q++) {
          for (int l = 0; l < ne; l++) {
            printf("%lf ", GdQ(q, l));
          }
          printf("\n");
        }
        printf("Permutation:\n");
        for (int q = 0; q < ne; q++) {
          printf("%i ", perm(q));
        }
        printf("\n");
        /////////////////////////////////////////////////////////////////////////////

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
        // Because L(0,0) == 1, we can just set the first element of z

        for (int l = 0; l < ne; l++) {
          for (int q = 0; q < l; q++) {
            tempRow(l) -= GdQ(l, q) * tempRow(q);
          }
        }

        /////////////////////////////////////////////////////////////////////////////
        printf("z:\n");
        for (int l = 0; l < ne; l++) {
          printf("%f ", tempRow(l));
        }
        printf("\n");
        /////////////////////////////////////////////////////////////////////////////

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

        /////////////////////////////////////////////////////////////////////////////
        printf("x:\n");
        for (int l = 0; l < ne; l++) {
          printf("%f ", b.dQ(i, j, k, l));
        }
        printf("\n");
        /////////////////////////////////////////////////////////////////////////////
      });
}

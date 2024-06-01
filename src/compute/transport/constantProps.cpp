#include "block_.hpp"
#include "compute.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"
#include <Kokkos_Core.hpp>
#include <math.h>

void constantProps(block_ &b, const thtrdat_ &th, const int &nface,
                   const int &indxI /*=0*/, const int &indxJ /*=0*/,
                   const int &indxK /*=0*/) {

#ifndef NSCOMPILE
  Kokkos::Experimental::UniqueToken<execSpace> token;
  int numIds = token.size();
  const int ns = th.ns;
  twoDview Y("Y", numIds, ns);
  twoDview X("X", numIds, ns);
#endif

#ifdef NSCOMPILE
#define Y(INDEX) Y[INDEX]
#define X(INDEX) X[INDEX]
#define ns NS
#else
#define Y(INDEX) Y(id, INDEX)
#define X(INDEX) X(id, INDEX)
#endif

  MDRange3 range = getRange3(b, nface, indxI, indxJ, indxK);
  Kokkos::parallel_for(
      "Const Props Transport", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
#ifndef NSCOMPILE
        int id = token.acquire();
#endif

#ifdef NSCOMPILE
        double Y(ns);
        double X(ns);
#endif

        // Compute nth species Y
        Y(ns - 1) = 1.0;
        for (int n = 0; n < ns - 1; n++) {
          Y(n) = b.q(i, j, k, 5 + n);
          Y(ns - 1) -= Y(n);
        }

        // Update mixture properties
        // Mole fractions
        {
          double mass = 0.0;
          for (int n = 0; n <= ns - 1; n++) {
            mass += Y(n) / th.MW(n);
          }
          for (int n = 0; n <= ns - 1; n++) {
            X(n) = Y(n) / th.MW(n) / mass;
          }
        }

        // viscosity mixture
        double mu = 0.0;
        for (int n = 0; n <= ns - 1; n++) {
          double phitemp = 0.0;
          for (int n2 = 0; n2 <= ns - 1; n2++) {
            double phi = pow((1.0 + sqrt(th.mu0(n) / th.mu0(n2) *
                                         sqrt(th.MW(n2) / th.MW(n)))),
                             2.0) /
                         (sqrt(8.0) * sqrt(1 + th.MW(n) / th.MW(n2)));
            phitemp += phi * X(n2);
          }
          mu += th.mu0(n) * X(n) / phitemp;
        }

        // thermal conductivity mixture
        double kappa;
        {
          double sum1 = 0.0;
          double sum2 = 0.0;
          for (int n = 0; n <= ns - 1; n++) {
            sum1 += X(n) * th.kappa0(n);
            sum2 += X(n) / th.kappa0(n);
          }
          kappa = 0.5 * (sum1 + 1.0 / sum2);
        }

        // Set values of new properties
        // viscocity
        b.qt(i, j, k, 0) = mu;
        // thermal conductivity
        b.qt(i, j, k, 1) = kappa;
        // Diffusion coefficients mass
        // NOTE: Unity Lewis number approximation!
        for (int n = 0; n <= ns - 1; n++) {
          b.qt(i, j, k, 2 + n) = kappa / (b.Q(i, j, k, 0) * b.qh(i, j, k, 1));
        }

#ifndef NSCOMPILE
        token.release(id);
#endif
      });
}

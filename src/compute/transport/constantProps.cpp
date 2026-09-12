#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>
#include <math.h>

PG_ABI void pgConstantProps(const pgView *Q_, const pgView *q_,
                            const pgView *qh_, const pgView *qt_,
                            const pgView *MW_, const pgView *kappa0_,
                            const pgView *lewis_, const pgView *mu0_, double Ru,
                            const pgRange *r) {
  auto Q = as4(*Q_);
  auto q = as4(*q_);
  auto qh = as4(*qh_);
  auto qt = as4(*qt_);
  auto MW = as1(*MW_);
  auto kappa0 = as1(*kappa0_);
  auto lewis = as1(*lewis_);
  auto mu0 = as1(*mu0_);
  const int ns = MW.extent(0);

#ifndef NSCOMPILE
  Kokkos::Experimental::UniqueToken<execSpace> token;
  int numIds = token.size();
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

  MDRange3 range = range3(*r);
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
          Y(n) = q(i, j, k, 5 + n);
          Y(ns - 1) -= Y(n);
        }

        // Update mixture properties
        // Mole fractions
        {
          double mass = 0.0;
          for (int n = 0; n <= ns - 1; n++) {
            mass += Y(n) / MW(n);
          }
          for (int n = 0; n <= ns - 1; n++) {
            X(n) = Y(n) / MW(n) / mass;
          }
        }

        // viscosity mixture
        double mu = 0.0;
        for (int n = 0; n <= ns - 1; n++) {
          double phitemp = 0.0;
          for (int n2 = 0; n2 <= ns - 1; n2++) {
            double phi =
                pow((1.0 + sqrt(mu0(n) / mu0(n2) * sqrt(MW(n2) / MW(n)))),
                    2.0) /
                (sqrt(8.0) * sqrt(1 + MW(n) / MW(n2)));
            phitemp += phi * X(n2);
          }
          mu += mu0(n) * X(n) / phitemp;
        }

        // thermal conductivity mixture
        double kappa;
        {
          double sum1 = 0.0;
          double sum2 = 0.0;
          for (int n = 0; n <= ns - 1; n++) {
            sum1 += X(n) * kappa0(n);
            sum2 += X(n) / kappa0(n);
          }
          kappa = 0.5 * (sum1 + 1.0 / sum2);
        }

        // Set values of new properties
        // viscocity
        qt(i, j, k, 0) = mu;
        // thermal conductivity
        qt(i, j, k, 1) = kappa;
        // Diffusion coefficients mass
        // NOTE: Unity Lewis number approximation!
        for (int n = 0; n <= ns - 1; n++) {
          qt(i, j, k, 2 + n) =
              kappa / (Q(i, j, k, 0) * qh(i, j, k, 1) * lewis(n));
        }

#ifndef NSCOMPILE
        token.release(id);
#endif
      });
}

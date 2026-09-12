#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>
#include <math.h>

PG_ABI void
pgKineticTheoryUnityLewis(const pgView *Q_, const pgView *q_, const pgView *qh_,
                          const pgView *qt_, const pgView *MW_,
                          const pgView *kappaPoly_, const pgView *lewis_,
                          const pgView *muPoly_, double Ru, const pgRange *r) {
  auto Q = as4(*Q_);
  auto q = as4(*q_);
  auto qh = as4(*qh_);
  auto qt = as4(*qt_);
  auto MW = as1(*MW_);
  auto kappaPoly = as2(*kappaPoly_);
  auto lewis = as1(*lewis_);
  auto muPoly = as2(*muPoly_);
  const int ns = MW.extent(0);

#ifndef NSCOMPILE
  Kokkos::Experimental::UniqueToken<execSpace> token;
  int numIds = token.size();
  twoDview Y("Y", numIds, ns);
  twoDview X("X", numIds, ns);
  twoDview mu_sp("mu_sp", numIds, ns);
  twoDview kappa_sp("kappa_sp", numIds, ns);
#endif

#ifdef NSCOMPILE
#define Y(INDEX) Y[INDEX]
#define X(INDEX) X[INDEX]
#define mu_sp(INDEX) mu_sp[INDEX]
#define kappa_sp(INDEX) kappa_sp[INDEX]
#define ns NS
#else
#define Y(INDEX) Y(id, INDEX)
#define X(INDEX) X(id, INDEX)
#define mu_sp(INDEX) mu_sp(id, INDEX)
#define kappa_sp(INDEX) kappa_sp(id, INDEX)
#endif

  // poly'l degree

  MDRange3 range = range3(*r);
  Kokkos::parallel_for(
      "Kinetic theory unity lewis", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
#ifndef NSCOMPILE
        int id = token.acquire();
#endif

        double &T = q(i, j, k, 4);
#ifdef NSCOMPILE
        double Y(ns);
        double X(ns);
        double mu_sp(ns) = {};
        double kappa_sp(ns) = {};
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
          // Mean molecular weight, mole fraction
          for (int n = 0; n <= ns - 1; n++) {
            X(n) = Y(n) / MW(n) / mass;
          }
        }

        // Evaluate all property polynomials, Horner in u = ln T
        const double u = log(T);
        const double sqrt_T = sqrt(T);
        const double sqrtsqrt_T = sqrt(sqrt_T);
        for (int n = 0; n <= ns - 1; n++) {
          // the scratch persists between cells; Horner starts from zero
          mu_sp(n) = 0.0;
          kappa_sp(n) = 0.0;
          for (int m = muPoly.extent(1) - 1; m >= 0; m--)
            mu_sp(n) = mu_sp(n) * u + muPoly(n, m);
          for (int m = kappaPoly.extent(1) - 1; m >= 0; m--)
            kappa_sp(n) = kappa_sp(n) * u + kappaPoly(n, m);

          // Set to the correct dimensions
          // the fit is of sqrt(mu)/T^(1/4), so undo both
          mu_sp(n) *= sqrtsqrt_T;
          mu_sp(n) *= mu_sp(n);
          kappa_sp(n) *= sqrt_T;
        }

        // Now every species' property is computed, generate mixture values

        // viscosity mixture
        double mu = 0.0;
        for (int n = 0; n <= ns - 1; n++) {
          double phitemp = 0.0;
          for (int n2 = 0; n2 <= ns - 1; n2++) {
            double phi =
                pow((1.0 + sqrt(mu_sp(n) / mu_sp(n2) * sqrt(MW(n2) / MW(n)))),
                    2.0) /
                (sqrt(8.0) * sqrt(1 + MW(n) / MW(n2)));
            phitemp += phi * X(n2);
          }
          mu += mu_sp(n) * X(n) / phitemp;
        }

        // thermal conductivity mixture
        double kappa = 0.0;
        {
          double sum1 = 0.0;
          double sum2 = 0.0;
          for (int n = 0; n <= ns - 1; n++) {
            sum1 += X(n) * kappa_sp(n);
            sum2 += X(n) / kappa_sp(n);
          }
          kappa = 0.5 * (sum1 + 1.0 / sum2);
        }

        // Set values of new properties
        // viscocity
        qt(i, j, k, 0) = mu;
        // thermal conductivity
        qt(i, j, k, 1) = kappa;
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

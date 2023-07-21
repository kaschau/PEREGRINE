#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "compute.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"
#include <math.h>

void kineticTheory(block_ &b, const thtrdat_ &th, const int &nface,
                   const int &indxI /*=0*/, const int &indxJ /*=0*/,
                   const int &indxK /*=0*/) {

#ifndef NSCOMPILE
  Kokkos::Experimental::UniqueToken<execSpace> token;
  int numIds = token.size();
  const int ns = th.ns;
  twoDview Y("Y", numIds, ns);
  twoDview X("X", numIds, ns);
  twoDview mu_sp("mu_sp", numIds, ns);
  twoDview kappa_sp("kappa_sp", numIds, ns);
  threeDview Dij("Dij", numIds, ns, ns);
  twoDview D("D", numIds, ns);
#endif

#ifdef NSCOMPILE
#define Y(INDEX) Y[INDEX]
#define X(INDEX) X[INDEX]
#define mu_sp(INDEX) mu_sp[INDEX]
#define kappa_sp(INDEX) kappa_sp[INDEX]
#define Dij(INDEX, INDEX1) Dij[INDEX][INDEX1]
#define D(INDEX) D[INDEX]
#define ns NS
#else
#define Y(INDEX) Y(id, INDEX)
#define X(INDEX) X(id, INDEX)
#define mu_sp(INDEX) mu_sp(id, INDEX)
#define kappa_sp(INDEX) kappa_sp(id, INDEX)
#define Dij(INDEX, INDEX1) Dij(id, INDEX, INDEX1)
#define D(INDEX) D(id, INDEX)
#endif
  // poly'l degree
  const int deg = 4;

  MDRange3 range = getRange3(b, nface, indxI, indxJ, indxK);
  Kokkos::parallel_for(
      "Kinetic Theory trans props", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
#ifndef NSCOMPILE
        int id = token.acquire();
#endif

        double &p = b.q(i, j, k, 0);
        double &T = b.q(i, j, k, 4);
#ifdef NSCOMPILE
        double Y(ns);
        double X(ns);
        double mu_sp(ns) = {};
        double kappa_sp(ns) = {};
        double Dij(ns, ns) = {};
        double D(ns);
#endif

        // check for pure fluid
        int pure = -1;

        // Compute nth species Y
        Y(ns - 1) = 1.0;
        for (int n = 0; n < ns - 1; n++) {
          Y(n) = b.q(i, j, k, 5 + n);
          Y(ns - 1) -= Y(n);
        }
        Y(ns - 1) = fmax(0.0, Y(ns - 1));

        // Update mixture properties
        // Mole fractions
        double MWmix = 0.0;
        {
          double mass = 0.0;
          for (int n = 0; n <= ns - 1; n++) {
            mass += Y(n) / th.MW(n);
          }

          // Mean molecular weight, mole fraction
          for (int n = 0; n <= ns - 1; n++) {
            X(n) = Y(n) / th.MW(n) / mass;
            if (X(n) == 1.0) {
              pure = n;
            }
            MWmix += X(n) * th.MW(n);
          }
        }

        // Evaluate all property polynomials
        const double logT = log(T);
        const double sqrt_T = exp(0.5 * logT);
        double logT_n[deg + 1];
        logT_n[0] = 1.0;
        for (int ply = 1; ply <= deg; ply++) {
          logT_n[ply] = logT * logT_n[ply - 1];
        }
        for (int n = 0; n <= ns - 1; n++) {
          // Evaluate polynomial
          for (int ply = 0; ply <= deg; ply++) {
            mu_sp(n) += th.muPoly(n, ply) * logT_n[ply];
            kappa_sp(n) += th.kappaPoly(n, ply) * logT_n[ply];

            for (int n2 = n; n2 <= ns - 1; n2++) {
              int indx = ns * (ns - 1) / 2 - (ns - n) * (ns - n - 1) / 2 + n2;
              Dij(n, n2) += th.DijPoly(indx, ply) * logT_n[ply];
            }
          }

          // Set to the correct dimensions
          mu_sp(n) *= sqrt_T;
          kappa_sp(n) *= sqrt_T;
          const double T_3o2 = pow(T, 1.5);
          for (int n2 = n; n2 <= ns - 1; n2++) {
            Dij(n, n2) *= T_3o2;
            Dij(n2, n) = Dij(n, n2);
          }
        }

        // Now every species' property is computed, generate mixture values

        // viscosity mixture
        double mu = 0.0;
        for (int n = 0; n <= ns - 1; n++) {
          double phitemp = 0.0;
          for (int n2 = 0; n2 <= ns - 1; n2++) {
            double phi = pow((1.0 + sqrt(mu_sp(n) / mu_sp(n2) *
                                         sqrt(th.MW(n2) / th.MW(n)))),
                             2.0) /
                         (sqrt(8.0) * sqrt(1.0 + th.MW(n) / th.MW(n2)));
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

        // mass diffusion coefficient mixture
        if (pure == -1) {
          for (int n = 0; n <= ns - 1; n++) {
            double sum1 = 0.0;
            double sum2 = 0.0;
            for (int n2 = 0; n2 <= ns - 1; n2++) {
              if (n == n2) {
                continue;
              }
              sum1 += X(n2) / Dij(n, n2);
              sum2 += X(n2) * th.MW(n2) / Dij(n, n2);
            }
            // Account for pressure
            sum1 *= p;
            sum2 *= p * X(n) / (MWmix - th.MW(n) * X(n));
            D(n) = 1.0 / (sum1 + sum2);
          }
        }

        // Set values of new properties
        // viscocity
        b.qt(i, j, k, 0) = mu;
        // thermal conductivity
        b.qt(i, j, k, 1) = kappa;
        // Diffusion coefficients mass
        for (int n = 0; n <= ns - 1; n++) {
          b.qt(i, j, k, 2 + n) = D(n);
        }

#ifndef NSCOMPILE
        token.release(id);
#endif
      });

  // if ns == 1, we need the diffusion coeff to be zero
  // so the viscous flux correction term is zero in
  // diffusiveFlux.cpp
  if (ns == 1) {
    MDRange3 range = getRange3(b, nface, indxI, indxJ, indxK);
    Kokkos::parallel_for(
        "Const Props Transport", range,
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
          b.qt(i, j, k, 2) = 0.0;
        });
  }
}

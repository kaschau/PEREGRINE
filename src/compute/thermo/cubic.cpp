#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "compute.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"
#include <math.h>
#include <stdexcept>

// References
//
// Generalizing the Thermodynamics State Relationships in KIVA-3V
//     Mario F. Trujillo
//     Peter O’Rourke
//     David Torres
//     Los Alamos National Labs, 2002
//     https://www.osti.gov/servlets/purl/809947
//
// Thermodyanamic Properties from Cubic Equations of State
//     Patrick Chung-Nin Mak
//     University of British Columbia, 1988
//     https://open.library.ubc.ca/media/download/pdf/831/1.0058883/2
//
// Thermal and Transport Properties for the Simulation of Direct-Fired sCO 2
// Combustor
//     Manikantachari, Martin, Bobren-Diaz, Vasu
//     Journal of Engineering for Gas Turbines and Power, 2017
//     DOI: 10.1Real Gas Models in Coupled Algorithms Numerical
//

// Real Gas Models in Coupled Algorithms Numerical Recipes and Thermophysical
// Relations
//     Hanimann, Mangani, Casartelli, Vogt, Darwish
//     Turbomachinery Propulsion and Power, 2020
//     doi:10.3390/ijtpp5030020

// ----------------------------------------------------------------------//
//           Solves the cubic EOS of the general form
//
//                    RT         a \alpha(T)
//               P = ____   _   _______________
//
//                   V-b          V^2+2bV-b^2
//
// ----------------------------------------------------------------------//

void cubic(block_ &b, const thtrdat_ &th, const int &nface,
           const std::string &given, const int &indxI /*=0*/,
           const int &indxJ /*=0*/, const int &indxK /*=0*/) {

#ifndef NSCOMPILE
  Kokkos::Experimental::UniqueToken<execSpace> token;
  int numIds = token.size();
  const int ns = th.ns;
  twoDview Y("Y", numIds, ns);
  twoDview hi("hi", numIds, ns);
  twoDview X("X", numIds, ns);
  twoDview ai("ai", numIds, ns);
#endif

#ifdef NSCOMPILE
#define Y(INDEX) Y[INDEX]
#define hi(INDEX) hi[INDEX]
#define X(INDEX) X[INDEX]
#define ai(INDEX) ai[INDEX]
#define ns NS
#else
#define Y(INDEX) Y(id, INDEX)
#define hi(INDEX) hi(id, INDEX)
#define X(INDEX) X(id, INDEX)
#define ai(INDEX) ai(id, INDEX)
#endif

  MDRange3 range = getRange3(b, nface, indxI, indxJ, indxK);
  if (given.compare("prims") == 0) {
    Kokkos::parallel_for(
        "Compute all conserved quantities from primatives via real gas", range,
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
#ifndef NSCOMPILE
          int id = token.acquire();
#endif

          // Updates all conserved quantities from primatives
          // Along the way, we need to compute mixture properties
          // gamma, cp, h, e, hi
          // So we store these as well.

          const double p = b.q(i, j, k, 0);
          const double u = b.q(i, j, k, 1);
          const double v = b.q(i, j, k, 2);
          const double w = b.q(i, j, k, 3);
          const double T = b.q(i, j, k, 4);
#ifdef NSCOMPILE
          double Y(ns);
          double X(ns);
#endif

          double rho;
          double rhou, rhov, rhow;
          double e, tke, rhoE;
          double gamma, cp, h, c;
#ifdef NSCOMPILE
          double hi(ns);
#endif
          double Rmix;

          // Compute nth species Y
          Y(ns - 1) = 1.0;
          double testSum = 0.0;
          // Compute y->x denom
          double denom = 0.0;
          for (int n = 0; n < ns - 1; n++) {
            b.q(i, j, k, 5 + n) = fmax(fmin(b.q(i, j, k, 5 + n), 1.0), 0.0);
            Y(n) = b.q(i, j, k, 5 + n);
            Y(ns - 1) -= Y(n);
            denom += Y(n) / th.MW(n);
            testSum += Y(n);
          }
          denom += Y(ns - 1) / th.MW(ns - 1);

          // Renormalize if necessary
          if (testSum > 1.0) {
            Y(ns - 1) = 0.0;
            for (int n = 0; n < ns - 1; n++) {
              Y(n) /= testSum;
            }
          }

          // Compute mole fraction, mean molecular weight
          double MWmix = 0.0;
          for (int n = 0; n <= ns - 1; n++) {
            X(n) = (Y(n) / th.MW(n)) / denom;
            MWmix += th.MW(n) * X(n);
          }
          // Compute Rmix
          Rmix = th.Ru / MWmix;

          // Real gas coefficients for cubic EOS
          // -------------------------------------------------------------------------------------------------------------//
          // Peng-Robinson
          const double uRG = 2.0, wRG = -1.0, biConst = 0.077796,
                       aiConst = 0.457240, fw0 = 0.37464, fw1 = 1.54226,
                       fw2 = -0.26992;
          // -------------------------------------------------------------------------------------------------------------//

          // -------------------------------------------------------------------------------------------------------------//
          // Soave-Redlich-Kwong
          // const double uRG=1.0, wRG= 0.0, biConst=0.0866403,
          // aiConst=0.4274802, fw0=0.480  , fw1=1.574  , fw2=-0.176  ;
          // -------------------------------------------------------------------------------------------------------------//

          double am = 0.0, bm = 0.0;
#ifdef NSCOMPILE
          double ai(ns);
#endif
          double Astar, Bstar;

          // Compressibility Factor, Z
          for (int n = 0; n <= ns - 1; n++) {
            double Tr = T / th.Tcrit(n);
            double fOmega =
                fw0 + fw1 * th.acentric(n) + fw2 * pow(th.acentric(n), 2.0);
            double alpha = pow(1.0 + fOmega * (1 - sqrt(Tr)), 2.0);
            ai(n) =
                aiConst * (pow(th.Ru * th.Tcrit(n), 2.0) * alpha) / th.pcrit(n);
            double bi = biConst * (th.Ru * th.Tcrit(n)) / th.pcrit(n);

            bm += X(n) * bi;
          }
          for (int n = 0; n <= ns - 1; n++) {
            for (int n2 = 0; n2 <= ns - 1; n2++) {
              am += X(n) * X(n2) *
                    sqrt(ai(n) *
                         ai(n2)); // - (1 - kij)  <- For now we ignore binary
                                  // interaciton coeff, i.e. assume kij=1
            }
          }

          Astar = am * p / pow(th.Ru * T, 2.0);
          Bstar = bm * p / (th.Ru * T);

          // Solve cubic EOS for Z
          // https://www.e-education.psu.edu/png520/m11_p6.html
          double z0, z1, z2, Z;
          double Bstar2 = pow(Bstar, 2.0);
          z0 = -(Astar * Bstar + wRG * Bstar2 + wRG * Bstar2 * Bstar);
          z1 = Astar + wRG * Bstar2 - uRG * Bstar - uRG * Bstar2;
          z2 = -(1.0 + Bstar - uRG * Bstar);

          double Q, RR, M;
          Q = (pow(z2, 2.0) - 3.0 * z1) / 9.0;
          RR = (2.0 * pow(z2, 3.0) - 9.0 * z2 * z1 + 27.0 * z0) / 54.0;
          M = pow(RR, 2.0) - pow(Q, 3.0);

          double z2o3 = z2 / 3.0;
          if (M > 0.0) {
            double S = -RR / abs(RR) * pow(abs(RR) + sqrt(M), (1.0 / 3.0));
            Z = S + Q / S - z2o3;
          } else {
            double q1p5 = pow(Q, 1.5);
            double sqQ = sqrt(Q);
            double theta = acos(RR / q1p5);
            double x1 = -(2.0 * sqQ * cos(theta / 3.0)) - z2o3;
            double x2 =
                -(2.0 * sqQ * cos((theta + 2 * 3.14159265358979323846) / 3.0)) -
                z2o3;
            double x3 =
                -(2.0 * sqQ * cos((theta - 2 * 3.14159265358979323846) / 3.0)) -
                z2o3;

            Z = fmax(x1, fmax(x2, x3));
          }

          // Update mixture properties

          // departure functions
          double dam = 0.0;
          for (int n = 0; n <= ns - 1; n++) {
            for (int n2 = 0; n2 <= ns - 1; n2++) {
              double fOmegaN =
                  fw0 + fw1 * th.acentric(n) + fw2 * pow(th.acentric(n), 2.0);
              double fOmegaN2 =
                  fw0 + fw1 * th.acentric(n2) + fw2 * pow(th.acentric(n2), 2.0);
              dam += X(n2) * X(n) * 1.0 *
                     (fOmegaN2 * sqrt(ai(n) * th.Tcrit(n2) / th.pcrit(n2)) +
                      fOmegaN * sqrt(ai(n2) * th.Tcrit(n) / th.pcrit(n)));
            }
          }
          dam *= -0.5 * th.Ru * sqrt(aiConst / T);
          double dAstardT = -2.0 * (Astar / T) * (1.0 - 0.5 * (T / am) * dam);
          double dBstardT = -Bstar / T;

          double dz0 = -(Bstar * dAstardT +
                         (Astar + (2.0 * Bstar + 3.0 * pow(Bstar, 2.0)) * wRG) *
                             dBstardT);
          double dz1 =
              (dAstardT + (2.0 * Bstar * (wRG - uRG) - uRG) * dBstardT);
          double dz2 = -(1.0 - uRG) * dBstardT;

          double dZdT = -(dz2 * pow(Z, 2.0) + dz1 * Z + dz0) /
                        (3.0 * pow(Z, 2.0) + 2.0 * Z * z2 + z1);

          double Cuw = 1.0 / (bm * th.Ru * sqrt(pow(uRG, 2.0) - 4.0 * wRG));
          double ZoB = Z / Bstar;
          double logZoB =
              log((2.0 * ZoB + (uRG - sqrt(pow(uRG, 2.0) - 4.0 * wRG))) /
                  (2.0 * ZoB + (uRG + sqrt(pow(uRG, 2.0) - 4.0 * wRG))));

          double cpDep = Cuw * (am / T - dam) * logZoB * 0.5 * T / am * dam +
                         pow((pow(ZoB, 2.0) + uRG * ZoB + wRG) -
                                 (dam / (bm * th.Ru)) * (ZoB - 1.0),
                             2.0) /
                             (pow(pow(ZoB, 2.0) + uRG * ZoB + wRG, 2.0) -
                              (am / (bm * th.Ru * T)) * (2.0 * ZoB + uRG) *
                                  pow(ZoB - 1.0, 2.0)) -
                         1.0;

          double hDep = Cuw * (am / T - dam) * logZoB + (Z - 1.0);

          // Start h and cp as departure values
          h = th.Ru * T * hDep / MWmix;
          cp = th.Ru * cpDep / MWmix;
          // start scope of precomputed T**
          {
            double Tinv = 1.0 / T;
            double To2 = T / 2.0;
            double T2 = pow(T, 2);
            double T3 = pow(T, 3);
            double T4 = pow(T, 4);
            double T2o3 = T2 / 3.0;
            double T3o4 = T3 / 4.0;
            double T4o5 = T4 / 5.0;
            for (int n = 0; n <= ns - 1; n++) {
              int m = (T <= th.NASA7(n, 0)) ? 8 : 1;

              double cps = (th.NASA7(n, m + 0) + th.NASA7(n, m + 1) * T +
                            th.NASA7(n, m + 2) * T2 + th.NASA7(n, m + 3) * T3 +
                            th.NASA7(n, m + 4) * T4) *
                           th.Ru / th.MW(n);

              hi(n) = (th.NASA7(n, m + 0) + th.NASA7(n, m + 1) * To2 +
                       th.NASA7(n, m + 2) * T2o3 + th.NASA7(n, m + 3) * T3o4 +
                       th.NASA7(n, m + 4) * T4o5 + th.NASA7(n, m + 5) * Tinv) *
                      T * th.Ru / th.MW(n);

              cp += cps * Y(n);
              h += hi(n) * Y(n);
              // Add departure to individual hi
              hi(n) += th.Ru * T * hDep / th.MW(n);
            }
            // end scope of precomputed T**
          }

          // Compute density
          rho = p / (Z * Rmix * T);

          // Specific heat ratio
          double dAstardp = Astar / p;
          double dBstardp = Bstar / p;

          dz0 = -(Bstar * dAstardp +
                  (Astar + (2.0 * Bstar + 3.0 * pow(Bstar, 2.0)) * wRG) *
                      dBstardp);
          dz1 = (dAstardp + (2.0 * Bstar * (wRG - uRG) - uRG) * dBstardp);
          dz2 = -(1.0 - uRG) * dBstardp;

          double dZdp = -(dz2 * pow(Z, 2.0) + dz1 * Z + dz0) /
                        (3.0 * pow(Z, 2.0) + 2.0 * Z * z2 + z1);

          double drhodp = (rho / p) * (1.e0 - p * dZdp / Z);
          double drhodt = -(rho / T) * (1.e0 + T * dZdT / Z);
          gamma = drhodp / (drhodp - (T / cp) * pow(drhodt / rho, 2.0));

          // Mixture speed of sound
          c = sqrt(abs(gamma / drhodp));

          // Compute momentum
          rhou = rho * u;
          rhov = rho * v;
          rhow = rho * w;
          // Compuute TKE
          tke = 0.5 * (pow(u, 2.0) + pow(v, 2.0) + pow(w, 2.0)) * rho;

          // Compute internal, total, energy
          e = h - p / rho;
          rhoE = rho * e + tke;

          // Set values of new properties
          // Density
          b.Q(i, j, k, 0) = rho;
          // Momentum
          b.Q(i, j, k, 1) = rhou;
          b.Q(i, j, k, 2) = rhov;
          b.Q(i, j, k, 3) = rhow;
          // Total Energy
          b.Q(i, j, k, 4) = rhoE;
          // Species mass
          for (int n = 0; n < ns - 1; n++) {
            b.Q(i, j, k, 5 + n) = Y(n) * rho;
          }
          // gamma,cp,h,c,e,hi
          b.qh(i, j, k, 0) = gamma;
          b.qh(i, j, k, 1) = cp;
          b.qh(i, j, k, 2) = rho * h;
          b.qh(i, j, k, 3) = c;
          b.qh(i, j, k, 4) = rho * e;
          for (int n = 0; n <= ns - 1; n++) {
            b.qh(i, j, k, 5 + n) = hi(n);
          }

#ifndef NSCOMPILE
          token.release(id);
#endif
        });
  }

  else if (given.compare("cons") == 0) {
    Kokkos::parallel_for(
        "Compute primatives from conserved quantities via real gas", range,
        KOKKOS_LAMBDA(const int i, const int j, const int k) {

#ifndef NSCOMPILE
          int id = token.acquire();
#endif

          // Updates all primatives from conserved quantities
          // Along the way, we need to compute mixture properties
          // gamma, cp, h, e, hi
          // So we store these as well.

          const double rho = b.Q(i, j, k, 0);
          const double rhou = b.Q(i, j, k, 1);
          const double rhov = b.Q(i, j, k, 2);
          const double rhow = b.Q(i, j, k, 3);
          const double rhoE = b.Q(i, j, k, 4);

          double p;
          double e, tke;
          double T;
#ifdef NSCOMPILE
          double Y(ns);
          double X(ns);
#endif
          double gamma, cp, h, c;
#ifdef NSCOMPILE
          double hi(ns);
#endif
          double Rmix;

          // Compute TKE
          tke = 0.5 * (pow(rhou, 2.0) + pow(rhov, 2.0) + pow(rhow, 2.0)) / rho;

          // Compute species mass fraction
          Y(ns - 1) = 1.0;
          double testSum = 0.0;
          // Compute y->x denom
          double denom = 0.0;
          for (int n = 0; n < ns - 1; n++) {
            b.Q(i, j, k, 5 + n) =
                fmax(fmin(b.Q(i, j, k, 5 + n), b.Q(i, j, k, 0)), 0.0);
            Y(n) = b.Q(i, j, k, 5 + n) / b.Q(i, j, k, 0);
            Y(ns - 1) -= Y(n);
            denom += Y(n) / th.MW(n);
            testSum += Y(n);
          }
          denom += Y(ns - 1) / th.MW(ns - 1);

          // Renormalize if necessary
          if (testSum > 1.0) {
            Y(ns - 1) = 0.0;
            for (int n = 0; n < ns - 1; n++) {
              Y(n) /= testSum;
              b.Q(i, j, k, 5 + n) = Y(n) * b.Q(i, j, k, 0);
            }
          }

          // Internal energy
          e = (rhoE - tke) / rho;

          // Real gas coefficients for cubic EOS

          // -------------------------------------------------------------------------------------------------------------//
          // Peng-Robinson
          const double uRG = 2.0, wRG = -1.0, biConst = 0.077796,
                       aiConst = 0.457240, fw0 = 0.37464, fw1 = 1.54226,
                       fw2 = -0.26992;
          // -------------------------------------------------------------------------------------------------------------//

          // -------------------------------------------------------------------------------------------------------------//
          // Soave-Redlich-Kwong
          // const double uRG=1.0, wRG= 0.0, biConst=0.0866403,
          // aiConst=0.4274802, fw0=0.480  , fw1=1.574  , fw2=-0.176  ;
          // -------------------------------------------------------------------------------------------------------------//

          // Iterate on to find temperature
          int nitr = 0, maxitr = 100;
          double tol = 1e-8;
          double error = 1e100;

          // Compute mole fraction, mean molecular weight
          double MWmix = 0.0;
          for (int n = 0; n <= ns - 1; n++) {
            X(n) = (Y(n) / th.MW(n)) / denom;
            MWmix += th.MW(n) * X(n);
          }
          // Compute Rmix
          Rmix = th.Ru / MWmix;
          // molar volume
          double Vm = MWmix / rho;

          double Astar, Bstar;
          double dZdT;
          double Z, z0, z1, z2;
          // Newtons method to find T
          T = (b.q(i, j, k, 4) < 1.0) ? 300.0
                                      : b.q(i, j, k, 4); // Initial guess of T
          while ((abs(error) > tol) && (nitr < maxitr)) {
            // With a T, we can compute p
            double bi;
            double am = 0.0;
            double bm = 0.0;
#ifdef NSCOMPILE
            double ai(ns);
#endif
            for (int n = 0; n <= ns - 1; n++) {
              double Tr = T / th.Tcrit(n);
              double fOmega =
                  fw0 + fw1 * th.acentric(n) + fw2 * pow(th.acentric(n), 2.0);
              double alpha = pow(1.0 + fOmega * (1 - sqrt(Tr)), 2.0);
              ai(n) = aiConst * (pow(th.Ru * th.Tcrit(n), 2.0) * alpha) /
                      th.pcrit(n);
              bi = biConst * (th.Ru * th.Tcrit(n)) / th.pcrit(n);

              bm += X(n) * bi;
            }
            for (int n = 0; n <= ns - 1; n++) {
              for (int n2 = 0; n2 <= ns - 1; n2++) {
                am += X(n) * X(n2) *
                      sqrt(ai(n) *
                           ai(n2)); // - (1 - kij)  <- For now we ignore binary
                                    // interaciton coeff, i.e. assume kij=1
              }
            }
            // PR
            double Cc = bm;
            // SRK
            // double Cc = 0.0;
            p = th.Ru * T / (Vm - bm) - am / (Vm * (Vm + bm) + Cc * (Vm - bm));

            Astar = am * p / pow(th.Ru * T, 2.0);
            Bstar = bm * p / (th.Ru * T);

            // Solve cubic EOS for Z
            // https://www.e-education.psu.edu/png520/m11_p6.html
            double Bstar2 = pow(Bstar, 2.0);
            z0 = -(Astar * Bstar + wRG * Bstar2 + wRG * Bstar2 * Bstar);
            z1 = Astar + wRG * Bstar2 - uRG * Bstar - uRG * Bstar2;
            z2 = -(1.0 + Bstar - uRG * Bstar);

            double Q, RR, M;
            Q = (pow(z2, 2.0) - 3.0 * z1) / 9.0;
            RR = (2.0 * pow(z2, 3.0) - 9.0 * z2 * z1 + 27.0 * z0) / 54.0;
            M = pow(RR, 2.0) - pow(Q, 3.0);

            double z2o3 = z2 / 3.0;
            if (M > 0.0) {
              double S = -RR / abs(RR) * pow(abs(RR) + sqrt(M), (1.0 / 3.0));
              Z = S + Q / S - z2o3;
            } else {
              double q1p5 = pow(Q, 1.5);
              double sqQ = sqrt(Q);
              double theta = acos(RR / q1p5);
              double x1 = -(2.0 * sqQ * cos(theta / 3.0)) - z2o3;
              double x2 = -(2.0 * sqQ *
                            cos((theta + 2 * 3.14159265358979323846) / 3.0)) -
                          z2o3;
              double x3 = -(2.0 * sqQ *
                            cos((theta - 2 * 3.14159265358979323846) / 3.0)) -
                          z2o3;

              Z = fmax(x1, fmax(x2, x3));
            }
            // departure functions
            double dam = 0.0;
            for (int n = 0; n <= ns - 1; n++) {
              for (int n2 = 0; n2 <= ns - 1; n2++) {
                double fOmegaN =
                    fw0 + fw1 * th.acentric(n) + fw2 * pow(th.acentric(n), 2.0);
                double fOmegaN2 = fw0 + fw1 * th.acentric(n2) +
                                  fw2 * pow(th.acentric(n2), 2.0);
                dam += X(n2) * X(n) * 1.0 *
                       (fOmegaN2 * sqrt(ai(n) * th.Tcrit(n2) / th.pcrit(n2)) +
                        fOmegaN * sqrt(ai(n2) * th.Tcrit(n) / th.pcrit(n)));
              }
            }
            dam *= -0.5 * th.Ru * sqrt(aiConst / T);
            double dAstardT = -2.0 * (Astar / T) * (1.0 - 0.5 * (T / am) * dam);
            double dBstardT = -Bstar / T;

            double dz0 =
                -(Bstar * dAstardT +
                  (Astar + (2.0 * Bstar + 3.0 * pow(Bstar, 2.0)) * wRG) *
                      dBstardT);
            double dz1 =
                (dAstardT + (2.0 * Bstar * (wRG - uRG) - uRG) * dBstardT);
            double dz2 = -(1.0 - uRG) * dBstardT;

            dZdT = -(dz2 * pow(Z, 2.0) + dz1 * Z + dz0) /
                   (3.0 * pow(Z, 2.0) + 2.0 * Z * z2 + z1);

            double Cuw = 1.0 / (bm * th.Ru * sqrt(pow(uRG, 2.0) - 4.0 * wRG));
            double ZoB = Z / Bstar;
            double logZoB =
                log((2.0 * ZoB + (uRG - sqrt(pow(uRG, 2.0) - 4.0 * wRG))) /
                    (2.0 * ZoB + (uRG + sqrt(pow(uRG, 2.0) - 4.0 * wRG))));

            double cpDep = Cuw * (am / T - dam) * logZoB * 0.5 * T / am * dam +
                           pow((pow(ZoB, 2.0) + uRG * ZoB + wRG) -
                                   (dam / (bm * th.Ru)) * (ZoB - 1.0),
                               2.0) /
                               (pow(pow(ZoB, 2.0) + uRG * ZoB + wRG, 2.0) -
                                (am / (bm * th.Ru * T)) * (2.0 * ZoB + uRG) *
                                    pow(ZoB - 1.0, 2.0)) -
                           1.0;

            double hDep = Cuw * (am / T - dam) * logZoB + (Z - 1.0);

            // Start h and cp as departure values
            h = th.Ru * T * hDep / MWmix;
            cp = th.Ru * cpDep / MWmix;
            // start scope of precomputed T**
            {
              double Tinv = 1.0 / T;
              double To2 = T / 2.0;
              double T2 = pow(T, 2);
              double T3 = pow(T, 3);
              double T4 = pow(T, 4);
              double T2o3 = T2 / 3.0;
              double T3o4 = T3 / 4.0;
              double T4o5 = T4 / 5.0;
              for (int n = 0; n <= ns - 1; n++) {
                int m = (T <= th.NASA7(n, 0)) ? 8 : 1;

                double cps =
                    (th.NASA7(n, m + 0) + th.NASA7(n, m + 1) * T +
                     th.NASA7(n, m + 2) * T2 + th.NASA7(n, m + 3) * T3 +
                     th.NASA7(n, m + 4) * T4) *
                    th.Ru / th.MW(n);

                hi(n) =
                    (th.NASA7(n, m + 0) + th.NASA7(n, m + 1) * To2 +
                     th.NASA7(n, m + 2) * T2o3 + th.NASA7(n, m + 3) * T3o4 +
                     th.NASA7(n, m + 4) * T4o5 + th.NASA7(n, m + 5) * Tinv) *
                    T * th.Ru / th.MW(n);

                cp += cps * Y(n);
                h += hi(n) * Y(n);
                // Add departure to individual hi
                hi(n) += th.Ru * T * hDep / th.MW(n);
              }
              // end scope of precomputed T**
            }

            error = e - (h - Z * Rmix * T);
            T = T - error / (-cp + Rmix * (Z * dZdT));
            nitr += 1;
          }

          // Specific heat ratio
          double dAstardp = Astar / p;
          double dBstardp = Bstar / p;

          double dz0 = -(Bstar * dAstardp +
                         (Astar + (2.0 * Bstar + 3.0 * pow(Bstar, 2.0)) * wRG) *
                             dBstardp);
          double dz1 =
              (dAstardp + (2.0 * Bstar * (wRG - uRG) - uRG) * dBstardp);
          double dz2 = -(1.0 - uRG) * dBstardp;

          double dZdp = -(dz2 * pow(Z, 2.0) + dz1 * Z + dz0) /
                        (3.0 * pow(Z, 2.0) + 2.0 * Z * z2 + z1);

          double drhodp = (rho / p) * (1.e0 - p * dZdp / Z);
          double drhodt = -(rho / T) * (1.e0 + T * dZdT / Z);
          gamma = drhodp / (drhodp - (T / cp) * pow(drhodt / rho, 2.0));

          // Mixture speed of sound
          c = sqrt(abs(gamma / drhodp));

          // Set values of new properties
          // Pressure, temperature, Y
          b.q(i, j, k, 0) = p;
          b.q(i, j, k, 1) = rhou / rho;
          b.q(i, j, k, 2) = rhov / rho;
          b.q(i, j, k, 3) = rhow / rho;
          b.q(i, j, k, 4) = T;
          for (int n = 0; n < ns - 1; n++) {
            b.q(i, j, k, 5 + n) = Y(n);
          }
          // gamma,cp,h,c,e,hi
          b.qh(i, j, k, 0) = gamma;
          b.qh(i, j, k, 1) = cp;
          b.qh(i, j, k, 2) = rho * h;
          b.qh(i, j, k, 3) = c;
          b.qh(i, j, k, 4) = rho * e;
          for (int n = 0; n <= ns - 1; n++) {
            b.qh(i, j, k, 5 + n) = hi(n);
          }

#ifndef NSCOMPILE
          token.release(id);
#endif
        });
  } else {
    throw std::invalid_argument("Invalid given string in real gas.");
  }
}

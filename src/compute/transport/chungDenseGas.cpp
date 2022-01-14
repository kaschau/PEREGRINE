#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "thtrdat_.hpp"
#include "compute.hpp"
#include <math.h>

// References
//
// Generalized Multiparameter Correlation for Nonpolar and Polar Fluid Transport Properties
//     Ting-Horng Chung
//     Ajlan
//     Lee
//     Starling
//     Ind. Eng. Chem. Res. 1988, 27,671-679


void chungDenseGas(block_ b,
                     const thtrdat_ th,
                     const int nface,
                     const int indxI/*=0*/,
                     const int indxJ/*=0*/,
                     const int indxK/*=0*/) {

  MDRange3 range = get_range3(b, nface, indxI, indxJ, indxK);
  Kokkos::Experimental::UniqueToken<exec_space> token;
  int numIds = token.size();

  const int ns=th.ns;
  twoDview Y("Y", ns, numIds);
  twoDview X("X", ns, numIds);
  //viscosity
  twoDview mu_sp("mu_sp", ns, numIds);
  //thermal conductivity
  twoDview kappa_sp("kappa_sp", ns, numIds);
  // binary diffusion
  threeDview Dij("Dij", ns, ns, numIds);
  twoDview Dk("Dk", ns, numIds);

  Kokkos::parallel_for(
      "Compute transport properties mu,kappa,Dij Chung dense fluid "
      "correlation.",
      range, KOKKOS_LAMBDA(const int i, const int j, const int k) {
        int id = token.acquire();

        double &T = b.q(i, j, k, 4);

        // Compute nth species Y
        Y(ns - 1, id) = 1.0;
        for (int n = 0; n < ns - 1; n++) {
          Y(n, id) = b.q(i, j, k, 5 + n);
          Y(ns - 1, id) -= Y(n, id);
        }
        Y(ns - 1, id) = fmax(0.0, Y(ns - 1, id));

        // Update mixture properties
        // Mole fractions
        double mass = 0.0;
        for (int n = 0; n <= ns - 1; n++) {
          mass += Y(n, id) / th.MW(n);
        }

        // Mean molecular weight, mole fraction
        double MWmix = 0.0;
        for (int n = 0; n <= ns - 1; n++) {
          X(n, id) = Y(n, id) / th.MW(n) / mass;
          MWmix += X(n, id) * th.MW(n);
        }

        double omegaStar, Tstar, Tr, Fc;
        double nu0;
        double Psi, alpha, beta, Z, cv;
        double lambda0;

        double Yy, G1, G2, nuk, nup;
        double H2, lambdak, lambdap;

        static constexpr double Acoeff = 1.16145;
        static constexpr double Bcoeff = 0.14874;
        static constexpr double Ccoeff = 0.52487;
        static constexpr double Dcoeff = 0.77320;
        static constexpr double Ecoeff = 2.16178;
        static constexpr double Fcoeff = 2.43787;
        static constexpr double Gcoeff = -6.435e-4;
        static constexpr double Hcoeff = 7.27371;
        static constexpr double Scoeff = 18.0323;
        static constexpr double Wcoeff = -0.76830;

        for (int n = 0; n <= ns - 1; n++) {
          // Modified Chapman-Enskog dulite gas

          // In the Chung paper, Vcrit is in cm^3/mol, but we store Vcrit in m^3/kg
          // TODO: Check units

          // Dilute gas viscosity
          Tr = T/th.Tcrit(n);
          Tstar = 1.2593*Tr;
          omegaStar = Acoeff/pow(Tstar,Bcoeff) + Ccoeff/exp(Dcoeff*Tstar) +
                      Ecoeff/exp(Fcoeff*Tstar) +
                      Gcoeff*pow(Tstar,Bcoeff)*sin(Scoeff*pow(Tstar,Wcoeff)-Hcoeff);


          Fc = 1 - 0.2756*th.acentric(n) + 0.059035*pow(th.redDipole(n),4.0); // + kij??

          nu0 = 4.0785e-5 * sqrt(th.MW(n)*T)/(pow(th.Vcrit(n),2.0/3.0)*omegaStar) * Fc;

          // Dilute gas thermal conductivity
          cv = b.qh(i,j,k,1)/b.qh(i,j,k,0);
          alpha = cv/th.Ru - 3.0/2.0;
          beta = 0.7862-0.7109*th.acentric(n)+1.3168*pow(th.acentric(n),2.0);
          Z = 2.0 + 10.5*pow(Tr,2.0);
          Psi = 1.0 + alpha*((0.215 + 0.28288*alpha - 1.061*beta + 0.26665*Z) /
                             (0.6366 + beta*Z + 1.061*alpha*beta));
          lambda0 = 7.452*nu0/th.MW(n)*Psi;


          // Viscosity for dense fluids
          //TODO: Move the Ai and Bi coefficients into python and do those calculations
          // before hand.
          double &A1 = th.chungA(n,0);
          double &A2 = th.chungA(n,1);
          double &A3 = th.chungA(n,2);
          double &A4 = th.chungA(n,3);
          double &A5 = th.chungA(n,4);
          double &A6 = th.chungA(n,5);
          double &A7 = th.chungA(n,6);
          double &A8 = th.chungA(n,7);
          double &A9 = th.chungA(n,8);
          double &A10 = th.chungA(n,9);

          Yy = b.Q(i,j,k,0)*th.Vcrit(n)/6.0;
          G1 =  (1.0-0.5*Yy)/pow(1-Yy,3.0);
          G2 = (A1*(1-exp(-A4*Yy)/Yy) +
                A2*G1*exp(A5*Yy) +
                A3*G1) / (A1*A4 + A2 + A3);
          nuk = nu0*(1/G2 + A6*Yy);
          nup = (36.344e-6 * sqrt(th.MW(n)*th.Tcrit(n))/pow(th.Vcrit(n),2.0/3.0))*
                 A7*pow(Yy,2.0)*G2*exp(A8)+A9/Tstar+A10/pow(Tstar,2.0);

          mu_sp(n,id) = nuk + nup;

          // Dilute thermal conductivity, in cal/(cm.s.K) so need to convert
          double &B1 = th.chungB(n,0);
          double &B2 = th.chungB(n,1);
          double &B3 = th.chungB(n,2);
          double &B4 = th.chungB(n,3);
          double &B5 = th.chungB(n,4);
          double &B6 = th.chungB(n,5);
          double &B7 = th.chungB(n,6);

          H2 = (B1*(1.0 - exp(-B4*Yy))/Yy +
                B2*G1*exp(B5*Yy) +
                B3*G1) / (B1*B4+B2+B3);

          lambdak = lambda0*(1.0/H2 + B6*Yy);
          lambdap = (3.039e-4*sqrt(th.Tcrit(n)/th.MW(n))/pow(th.Vcrit(n),2.0/3.0))*B7*pow(Yy,2.0)*H2*sqrt(Tr);

          kappa_sp(n,id) =  lambdak + lambdap;

        }

        // Now every species' property is computed, generate mixture values

        // viscosity mixture
        double phi;
        double mu = 0.0;
        for (int n = 0; n <= ns - 1; n++) {
          double phitemp = 0.0;
          for (int n2 = 0; n2 <= ns - 1; n2++) {
            phi = pow((1.0 + sqrt(mu_sp(n, id) / mu_sp(n2, id) *
                                  sqrt(th.MW(n2) / th.MW(n)))),
                      2.0) /
                  (sqrt(8.0) * sqrt(1 + th.MW(n) / th.MW(n2)));
            phitemp += phi * X(n2, id);
          }
          mu += mu_sp(n, id) * X(n, id) / phitemp;
        }

        // thermal conductivity mixture
        double kappa = 0.0;

        double sum1 = 0.0;
        double sum2 = 0.0;
        for (int n = 0; n <= ns - 1; n++) {
          sum1 += X(n, id) * kappa_sp(n, id);
          sum2 += X(n, id) / kappa_sp(n, id);
        }
        kappa = 0.5 * (sum1 + 1.0 / sum2);

        // // mass diffusion coefficient mixture
        // double temp;
        // for (int n = 0; n <= ns - 1; n++) {
        //   sum1 = 0.0;
        //   sum2 = 0.0;
        //   for (int n2 = 0; n2 <= ns - 1; n2++) {
        //     if (n == n2) {
        //       continue;
        //     }
        //     sum1 += X(n2, id) / Dij(n, n2, id);
        //     sum2 += X(n2, id) * th.MW(n2) / Dij(n, n2, id);
        //   }
        //   // Account for pressure
        //   sum1 *= p;
        //   // HACK must be a better way to give zero for sum2 when MWmix ==
        //   // th.MW(n)*X(n)
        //   temp = p * X(n, id) / (MWmix - th.MW(n) * X(n, id));
        //   if (isinf(temp)) {
        //     Dk(n, id) = 0.0;
        //   } else {
        //     sum2 *= temp;
        //     Dk(n, id) = 1.0 / (sum1 + sum2);
        //   }
        // }

        // Set values of new properties
        // viscocity
        b.qt(i, j, k, 0) = mu;
        // thermal conductivity
        b.qt(i, j, k, 1) = kappa;
        // // Diffusion coefficients mass
        // for (int n = 0; n <= ns - 1; n++) {
        //   b.qt(i, j, k, 2 + n) = Dk(n, id);
        // }

        token.release(id);
      });
}

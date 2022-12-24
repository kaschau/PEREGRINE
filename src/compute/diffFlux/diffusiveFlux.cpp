#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkosTypes.hpp"
#include "math.h"
#include "thtrdat_.hpp"

void diffusiveFlux(block_ &b) {

  //-------------------------------------------------------------------------------------------|
  // i flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_i({b.ng, b.ng, b.ng},
                   {b.ni + b.ng, b.nj + b.ng - 1, b.nk + b.ng - 1});
  Kokkos::parallel_for(
      "i face visc fluxes", range_i,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double txx, txy, txz;
        double tyx, tyy, tyz;
        double tzx, tzy, tzz;

        double uf, vf, wf;
        double q;

        const double c23 = 2.0 / 3.0;

        double mu = 0.5 * (b.qt(i, j, k, 0) + b.qt(i - 1, j, k, 0));
        double kappa = 0.5 * (b.qt(i, j, k, 1) + b.qt(i - 1, j, k, 1));

        // continuity
        b.iF(i, j, k, 0) = 0.0;

        // Derivatives on face
        double dudx = 0.5 * (b.dqdx(i, j, k, 1) + b.dqdx(i - 1, j, k, 1));
        double dvdx = 0.5 * (b.dqdx(i, j, k, 2) + b.dqdx(i - 1, j, k, 2));
        double dwdx = 0.5 * (b.dqdx(i, j, k, 3) + b.dqdx(i - 1, j, k, 3));

        double dudy = 0.5 * (b.dqdy(i, j, k, 1) + b.dqdy(i - 1, j, k, 1));
        double dvdy = 0.5 * (b.dqdy(i, j, k, 2) + b.dqdy(i - 1, j, k, 2));
        double dwdy = 0.5 * (b.dqdy(i, j, k, 3) + b.dqdy(i - 1, j, k, 3));

        double dudz = 0.5 * (b.dqdz(i, j, k, 1) + b.dqdz(i - 1, j, k, 1));
        double dvdz = 0.5 * (b.dqdz(i, j, k, 2) + b.dqdz(i - 1, j, k, 2));
        double dwdz = 0.5 * (b.dqdz(i, j, k, 3) + b.dqdz(i - 1, j, k, 3));

        // x momentum
        txx = c23 * mu * (2.0 * dudx - dvdy - dwdz);
        txy = mu * (dvdx + dudy);
        txz = mu * (dwdx + dudz);

        b.iF(i, j, k, 1) = -(txx * b.isx(i, j, k) + txy * b.isy(i, j, k) +
                             txz * b.isz(i, j, k));

        // y momentum
        tyx = txy;
        tyy = c23 * mu * (-dudx + 2.0 * dvdy - dwdz);
        tyz = mu * (dwdy + dvdz);

        b.iF(i, j, k, 2) = -(tyx * b.isx(i, j, k) + tyy * b.isy(i, j, k) +
                             tyz * b.isz(i, j, k));

        // z momentum
        tzx = txz;
        tzy = tyz;
        tzz = c23 * mu * (-dudx - dvdy + 2.0 * dwdz);

        b.iF(i, j, k, 3) = -(tzx * b.isx(i, j, k) + tzy * b.isy(i, j, k) +
                             tzz * b.isz(i, j, k));

        // energy
        //   heat conduction
        double dTdx = 0.5 * (b.dqdx(i, j, k, 4) + b.dqdx(i - 1, j, k, 4));
        double dTdy = 0.5 * (b.dqdy(i, j, k, 4) + b.dqdy(i - 1, j, k, 4));
        double dTdz = 0.5 * (b.dqdz(i, j, k, 4) + b.dqdz(i - 1, j, k, 4));

        q = -kappa * (dTdx * b.isx(i, j, k) + dTdy * b.isy(i, j, k) +
                      dTdz * b.isz(i, j, k));

        // flow work
        // Compute face normal volume flux vector
        uf = 0.5 * (b.q(i, j, k, 1) + b.q(i - 1, j, k, 1));
        vf = 0.5 * (b.q(i, j, k, 2) + b.q(i - 1, j, k, 2));
        wf = 0.5 * (b.q(i, j, k, 3) + b.q(i - 1, j, k, 3));

        b.iF(i, j, k, 4) = -(uf * txx + vf * txy + wf * txz) * b.isx(i, j, k) -
                           (uf * tyx + vf * tyy + wf * tyz) * b.isy(i, j, k) -
                           (uf * tzx + vf * tzy + wf * tzz) * b.isz(i, j, k) +
                           q;

        // Species
        double Dk, Vc = 0.0;
        double gradYns = 0.0;
        double rho = 0.5 * (b.Q(i, j, k, 0) + b.Q(i - 1, j, k, 0));
        // Compute the species flux and correction term \sum(k=1,ns) Dk*gradYk
        for (int n = 0; n < b.ne - 5; n++) {
          Dk = 0.5 * (b.qt(i, j, k, 2 + n) + b.qt(i - 1, j, k, 2 + n));
          double dYdx =
              0.5 * (b.dqdx(i, j, k, 5 + n) + b.dqdx(i - 1, j, k, 5 + n));
          double dYdy =
              0.5 * (b.dqdy(i, j, k, 5 + n) + b.dqdy(i - 1, j, k, 5 + n));
          double dYdz =
              0.5 * (b.dqdz(i, j, k, 5 + n) + b.dqdz(i - 1, j, k, 5 + n));

          double gradYk = (dYdx * b.isx(i, j, k) + dYdy * b.isy(i, j, k) +
                           dYdz * b.isz(i, j, k));
          gradYns -= gradYk;
          Vc += Dk * gradYk;
          b.iF(i, j, k, 5 + n) = -rho * Dk * gradYk;
        }
        // Apply n=ns species to correction
        Dk = 0.5 *
             (b.qt(i, j, k, 2 + b.ne - 5) + b.qt(i - 1, j, k, 2 + b.ne - 5));
        Vc += Dk * gradYns;

        // Apply correction and species thermal flux
        double Yk, hk;
        double Yns = 1.0;
        for (int n = 0; n < b.ne - 5; n++) {
          Yk = 0.5 * (b.q(i, j, k, 5 + n) + b.q(i - 1, j, k, 5 + n));
          Yns -= Yk;
          b.iF(i, j, k, 5 + n) += Yk * rho * Vc;
          // Species thermal diffusion
          hk = 0.5 * (b.qh(i, j, k, 5 + n) + b.qh(i - 1, j, k, 5 + n));
          b.iF(i, j, k, 4) += b.iF(i, j, k, 5 + n) * hk;
        }
        // Apply the n=ns species to thermal diffusion
        Yns = fmax(Yns, 0.0);
        hk = 0.5 * (b.qh(i, j, k, b.ne) + b.qh(i - 1, j, k, b.ne));
        b.iF(i, j, k, 4) += (-rho * Dk * gradYns + Yns * rho * Vc) * hk;
      });

  //-------------------------------------------------------------------------------------------|
  // j flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_j({b.ng, b.ng, b.ng},
                   {b.ni + b.ng - 1, b.nj + b.ng, b.nk + b.ng - 1});
  Kokkos::parallel_for(
      "j face visc fluxes", range_j,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double txx, txy, txz;
        double tyx, tyy, tyz;
        double tzx, tzy, tzz;

        double uf, vf, wf;
        double q;

        const double c23 = 2.0 / 3.0;

        double mu = 0.5 * (b.qt(i, j, k, 0) + b.qt(i, j - 1, k, 0));
        double kappa = 0.5 * (b.qt(i, j, k, 1) + b.qt(i, j - 1, k, 1));

        // continuity
        b.jF(i, j, k, 0) = 0.0;

        // Spatial derivative on face
        double dudx = 0.5 * (b.dqdx(i, j, k, 1) + b.dqdx(i, j - 1, k, 1));
        double dvdx = 0.5 * (b.dqdx(i, j, k, 2) + b.dqdx(i, j - 1, k, 2));
        double dwdx = 0.5 * (b.dqdx(i, j, k, 3) + b.dqdx(i, j - 1, k, 3));

        double dudy = 0.5 * (b.dqdy(i, j, k, 1) + b.dqdy(i, j - 1, k, 1));
        double dvdy = 0.5 * (b.dqdy(i, j, k, 2) + b.dqdy(i, j - 1, k, 2));
        double dwdy = 0.5 * (b.dqdy(i, j, k, 3) + b.dqdy(i, j - 1, k, 3));

        double dudz = 0.5 * (b.dqdz(i, j, k, 1) + b.dqdz(i, j - 1, k, 1));
        double dvdz = 0.5 * (b.dqdz(i, j, k, 2) + b.dqdz(i, j - 1, k, 2));
        double dwdz = 0.5 * (b.dqdz(i, j, k, 3) + b.dqdz(i, j - 1, k, 3));

        // x momentum
        txx = c23 * mu * (2.0 * dudx - dvdy - dwdz);
        txy = mu * (dvdx + dudy);
        txz = mu * (dwdx + dudz);

        b.jF(i, j, k, 1) = -(txx * b.jsx(i, j, k) + txy * b.jsy(i, j, k) +
                             txz * b.jsz(i, j, k));

        // y momentum
        tyx = txy;
        tyy = c23 * mu * (-dudx + 2.0 * dvdy - dwdz);
        tyz = mu * (dwdy + dvdz);

        b.jF(i, j, k, 2) = -(tyx * b.jsx(i, j, k) + tyy * b.jsy(i, j, k) +
                             tyz * b.jsz(i, j, k));

        // z momentum
        tzx = txz;
        tzy = tyz;
        tzz = c23 * mu * (-dudx - dvdy + 2.0 * dwdz);

        b.jF(i, j, k, 3) = -(tzx * b.jsx(i, j, k) + tzy * b.jsy(i, j, k) +
                             tzz * b.jsz(i, j, k));

        // energy
        //   heat conduction
        double dTdx = 0.5 * (b.dqdx(i, j, k, 4) + b.dqdx(i, j - 1, k, 4));
        double dTdy = 0.5 * (b.dqdy(i, j, k, 4) + b.dqdy(i, j - 1, k, 4));
        double dTdz = 0.5 * (b.dqdz(i, j, k, 4) + b.dqdz(i, j - 1, k, 4));

        q = -kappa * (dTdx * b.jsx(i, j, k) + dTdy * b.jsy(i, j, k) +
                      dTdz * b.jsz(i, j, k));

        // flow work
        // Compute face normal volume flux vector
        uf = 0.5 * (b.q(i, j, k, 1) + b.q(i, j - 1, k, 1));
        vf = 0.5 * (b.q(i, j, k, 2) + b.q(i, j - 1, k, 2));
        wf = 0.5 * (b.q(i, j, k, 3) + b.q(i, j - 1, k, 3));

        b.jF(i, j, k, 4) = -(uf * txx + vf * txy + wf * txz) * b.jsx(i, j, k) -
                           (uf * tyx + vf * tyy + wf * tyz) * b.jsy(i, j, k) -
                           (uf * tzx + vf * tzy + wf * tzz) * b.jsz(i, j, k) +
                           q;

        // Species
        double Dk, Vc = 0.0;
        double gradYns = 0.0;
        double rho = 0.5 * (b.Q(i, j, k, 0) + b.Q(i, j - 1, k, 0));
        // Compute the species flux and correction term \sum(k=1,ns) Dk*gradYk
        for (int n = 0; n < b.ne - 5; n++) {
          Dk = 0.5 * (b.qt(i, j, k, 2 + n) + b.qt(i, j - 1, k, 2 + n));
          double dYdx =
              0.5 * (b.dqdx(i, j, k, 5 + n) + b.dqdx(i, j - 1, k, 5 + n));
          double dYdy =
              0.5 * (b.dqdy(i, j, k, 5 + n) + b.dqdy(i, j - 1, k, 5 + n));
          double dYdz =
              0.5 * (b.dqdz(i, j, k, 5 + n) + b.dqdz(i, j - 1, k, 5 + n));

          double gradYk = (dYdx * b.jsx(i, j, k) + dYdy * b.jsy(i, j, k) +
                           dYdz * b.jsz(i, j, k));
          gradYns -= gradYk;
          Vc += Dk * gradYk;
          b.jF(i, j, k, 5 + n) = -rho * Dk * gradYk;
        }
        // Apply n=ns species to correction
        Dk = 0.5 *
             (b.qt(i, j, k, 2 + b.ne - 5) + b.qt(i, j - 1, k, 2 + b.ne - 5));
        Vc += Dk * gradYns;

        // Apply correction and species thermal flux
        double Yk, hk;
        double Yns = 1.0;
        for (int n = 0; n < b.ne - 5; n++) {
          Yk = 0.5 * (b.q(i, j, k, 5 + n) + b.q(i, j - 1, k, 5 + n));
          Yns -= Yk;
          b.jF(i, j, k, 5 + n) += Yk * rho * Vc;
          // Species thermal diffusion
          hk = 0.5 * (b.qh(i, j, k, 5 + n) + b.qh(i, j - 1, k, 5 + n));
          b.jF(i, j, k, 4) += b.jF(i, j, k, 5 + n) * hk;
        }
        // Apply the n=ns species to thermal diffusion
        Yns = fmax(Yns, 0.0);
        hk = 0.5 * (b.qh(i, j, k, b.ne) + b.qh(i, j - 1, k, b.ne));
        b.jF(i, j, k, 4) += (-rho * Dk * gradYns + Yns * rho * Vc) * hk;
      });

  //-------------------------------------------------------------------------------------------|
  // k flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_k({b.ng, b.ng, b.ng},
                   {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng});
  Kokkos::parallel_for(
      "k face visc fluxes", range_k,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double txx, txy, txz;
        double tyx, tyy, tyz;
        double tzx, tzy, tzz;

        double uf, vf, wf;
        double q;

        const double c23 = 2.0 / 3.0;

        double mu = 0.5 * (b.qt(i, j, k, 0) + b.qt(i, j, k - 1, 0));
        double kappa = 0.5 * (b.qt(i, j, k, 1) + b.qt(i, j, k - 1, 1));

        // continuity
        b.kF(i, j, k, 0) = 0.0;

        // Spatial derivative on face
        double dudx = 0.5 * (b.dqdx(i, j, k, 1) + b.dqdx(i, j, k - 1, 1));
        double dvdx = 0.5 * (b.dqdx(i, j, k, 2) + b.dqdx(i, j, k - 1, 2));
        double dwdx = 0.5 * (b.dqdx(i, j, k, 3) + b.dqdx(i, j, k - 1, 3));

        double dudy = 0.5 * (b.dqdy(i, j, k, 1) + b.dqdy(i, j, k - 1, 1));
        double dvdy = 0.5 * (b.dqdy(i, j, k, 2) + b.dqdy(i, j, k - 1, 2));
        double dwdy = 0.5 * (b.dqdy(i, j, k, 3) + b.dqdy(i, j, k - 1, 3));

        double dudz = 0.5 * (b.dqdz(i, j, k, 1) + b.dqdz(i, j, k - 1, 1));
        double dvdz = 0.5 * (b.dqdz(i, j, k, 2) + b.dqdz(i, j, k - 1, 2));
        double dwdz = 0.5 * (b.dqdz(i, j, k, 3) + b.dqdz(i, j, k - 1, 3));

        // x momentum
        txx = c23 * mu * (2.0 * dudx - dvdy - dwdz);
        txy = mu * (dvdx + dudy);
        txz = mu * (dwdx + dudz);

        b.kF(i, j, k, 1) = -(txx * b.ksx(i, j, k) + txy * b.ksy(i, j, k) +
                             txz * b.ksz(i, j, k));

        // y momentum
        tyx = txy;
        tyy = c23 * mu * (-dudx + 2.0 * dvdy - dwdz);
        tyz = mu * (dwdy + dvdz);

        b.kF(i, j, k, 2) = -(tyx * b.ksx(i, j, k) + tyy * b.ksy(i, j, k) +
                             tyz * b.ksz(i, j, k));

        // z momentum
        tzx = txz;
        tzy = tyz;
        tzz = c23 * mu * (-dudx - dvdy + 2.0 * dwdz);

        b.kF(i, j, k, 3) = -(tzx * b.ksx(i, j, k) + tzy * b.ksy(i, j, k) +
                             tzz * b.ksz(i, j, k));

        // energy
        //   heat conduction
        double dTdx = 0.5 * (b.dqdx(i, j, k, 4) + b.dqdx(i, j, k - 1, 4));
        double dTdy = 0.5 * (b.dqdy(i, j, k, 4) + b.dqdy(i, j, k - 1, 4));
        double dTdz = 0.5 * (b.dqdz(i, j, k, 4) + b.dqdz(i, j, k - 1, 4));

        q = -kappa * (dTdx * b.ksx(i, j, k) + dTdy * b.ksy(i, j, k) +
                      dTdz * b.ksz(i, j, k));

        // flow work
        // Compute face normal volume flux vector
        uf = 0.5 * (b.q(i, j, k, 1) + b.q(i, j, k - 1, 1));
        vf = 0.5 * (b.q(i, j, k, 2) + b.q(i, j, k - 1, 2));
        wf = 0.5 * (b.q(i, j, k, 3) + b.q(i, j, k - 1, 3));

        b.kF(i, j, k, 4) = -(uf * txx + vf * txy + wf * txz) * b.ksx(i, j, k) -
                           (uf * tyx + vf * tyy + wf * tyz) * b.ksy(i, j, k) -
                           (uf * tzx + vf * tzy + wf * tzz) * b.ksz(i, j, k) +
                           q;

        // Species
        double Dk, Vc = 0.0;
        double gradYns = 0.0;
        double rho = 0.5 * (b.Q(i, j, k, 0) + b.Q(i, j, k - 1, 0));
        // Compute the species flux and correction term \sum(k=1,ns) Dk*gradYk
        for (int n = 0; n < b.ne - 5; n++) {
          Dk = 0.5 * (b.qt(i, j, k, 2 + n) + b.qt(i, j, k - 1, 2 + n));
          double dYdx =
              0.5 * (b.dqdx(i, j, k, 5 + n) + b.dqdx(i, j, k - 1, 5 + n));
          double dYdy =
              0.5 * (b.dqdy(i, j, k, 5 + n) + b.dqdy(i, j, k - 1, 5 + n));
          double dYdz =
              0.5 * (b.dqdz(i, j, k, 5 + n) + b.dqdz(i, j, k - 1, 5 + n));
          double gradYk = (dYdx * b.ksx(i, j, k) + dYdy * b.ksy(i, j, k) +
                           dYdz * b.ksz(i, j, k));
          gradYns -= gradYk;
          Vc += Dk * gradYk;
          b.kF(i, j, k, 5 + n) = -rho * Dk * gradYk;
        }
        // Apply n=ns species to correction
        Dk = 0.5 *
             (b.qt(i, j, k, 2 + b.ne - 5) + b.qt(i, j, k - 1, 2 + b.ne - 5));
        Vc += Dk * gradYns;

        // Apply correction and species thermal flux
        double Yk, hk;
        double Yns = 1.0;
        for (int n = 0; n < b.ne - 5; n++) {
          Yk = 0.5 * (b.q(i, j, k, 5 + n) + b.q(i, j, k - 1, 5 + n));
          Yns -= Yk;
          b.kF(i, j, k, 5 + n) += Yk * rho * Vc;
          // Species thermal diffusion
          hk = 0.5 * (b.qh(i, j, k, 5 + n) + b.qh(i, j, k - 1, 5 + n));
          b.kF(i, j, k, 4) += b.kF(i, j, k, 5 + n) * hk;
        }
        // Apply the n=ns species to thermal diffusion
        Yns = fmax(Yns, 0.0);
        hk = 0.5 * (b.qh(i, j, k, b.ne) + b.qh(i, j, k - 1, b.ne));
        b.kF(i, j, k, 4) += (-rho * Dk * gradYns + Yns * rho * Vc) * hk;
      });
}

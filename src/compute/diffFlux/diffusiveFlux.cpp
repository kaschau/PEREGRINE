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
        double dudx, dudy, dudz;
        double dvdx, dvdy, dvdz;
        double dwdx, dwdy, dwdz;

        double txx, txy, txz;
        double tyx, tyy, tyz;
        double tzx, tzy, tzz;

        double dTdx, dTdy, dTdz;
        double uf, vf, wf;
        double q;

        double dYdx, dYdy, dYdz;

        const double c23 = 2.0 / 3.0;

        double mu = 0.5 * (b.qt(i, j, k, 0) + b.qt(i - 1, j, k, 0));
        double kappa = 0.5 * (b.qt(i, j, k, 1) + b.qt(i - 1, j, k, 1));

        // continuity
        b.iF(i, j, k, 0) = 0.0;

        // Derivatives on face
        dudx = b.idqdx(i, j, k, 1);
        dvdx = b.idqdx(i, j, k, 2);
        dwdx = b.idqdx(i, j, k, 3);

        dudy = b.idqdy(i, j, k, 1);
        dvdy = b.idqdy(i, j, k, 2);
        dwdy = b.idqdy(i, j, k, 3);

        dudz = b.idqdz(i, j, k, 1);
        dvdz = b.idqdz(i, j, k, 2);
        dwdz = b.idqdz(i, j, k, 3);

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
        dTdx = b.idqdx(i, j, k, 4);
        dTdy = b.idqdy(i, j, k, 4);
        dTdz = b.idqdz(i, j, k, 4);

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
        double gradYk, Dij, Dcorr = 0.0;
        double gradYns = 0.0;
        double rho = 0.5 * (b.Q(i, j, k, 0) + b.Q(i - 1, j, k, 0));
        // Compute the species flux and correction term \sum(k=1,ns) Dij*gradYk
        for (int n = 0; n < b.ne - 5; n++) {
          Dij = 0.5 * (b.qt(i, j, k, 2 + n) + b.qt(i - 1, j, k, 2 + n));
          dYdx = b.idqdx(i, j, k, 5 + n);
          dYdy = b.idqdy(i, j, k, 5 + n);
          dYdz = b.idqdz(i, j, k, 5 + n);
          gradYk = (dYdx * b.isx(i, j, k) + dYdy * b.isy(i, j, k) +
                    dYdz * b.isz(i, j, k));
          gradYns -= gradYk;
          Dcorr += Dij * gradYk;
          b.iF(i, j, k, 5 + n) = -rho * Dij * gradYk;
        }
        // Apply n=ns species to correction
        Dij = 0.5 *
              (b.qt(i, j, k, 2 + b.ne - 5) + b.qt(i - 1, j, k, 2 + b.ne - 5));
        Dcorr += Dij * gradYns;

        // Apply correction and species thermal flux
        double Yk, hk;
        double Yns = 1.0;
        for (int n = 0; n < b.ne - 5; n++) {
          Yk = 0.5 * (b.q(i, j, k, 5 + n) + b.q(i - 1, j, k, 5 + n));
          Yns -= Yk;
          b.iF(i, j, k, 5 + n) += Yk * rho * Dcorr;
          // Species thermal diffusion
          hk = 0.5 * (b.qh(i, j, k, 5 + n) + b.qh(i - 1, j, k, 5 + n));
          b.iF(i, j, k, 4) += b.iF(i, j, k, 5 + n) * hk;
        }
        // Apply the n=ns species to thermal diffusion
        Yns = fmax(Yns, 0.0);
        hk = 0.5 * (b.qh(i, j, k, b.ne) + b.qh(i - 1, j, k, b.ne));
        b.iF(i, j, k, 4) += (-rho * Dij * gradYns + Yns * rho * Dcorr) * hk;
      });

  //-------------------------------------------------------------------------------------------|
  // j flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_j({b.ng, b.ng, b.ng},
                   {b.ni + b.ng - 1, b.nj + b.ng, b.nk + b.ng - 1});
  Kokkos::parallel_for(
      "j face visc fluxes", range_j,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double dudx, dudy, dudz;
        double dvdx, dvdy, dvdz;
        double dwdx, dwdy, dwdz;

        double txx, txy, txz;
        double tyx, tyy, tyz;
        double tzx, tzy, tzz;

        double dTdx, dTdy, dTdz;
        double uf, vf, wf;
        double q;

        double dYdx, dYdy, dYdz;

        const double c23 = 2.0 / 3.0;

        double mu = 0.5 * (b.qt(i, j, k, 0) + b.qt(i, j - 1, k, 0));
        double kappa = 0.5 * (b.qt(i, j, k, 1) + b.qt(i, j - 1, k, 1));

        // continuity
        b.jF(i, j, k, 0) = 0.0;

        // Spatial derivative on face
        dudx = b.jdqdx(i, j, k, 1);
        dvdx = b.jdqdx(i, j, k, 2);
        dwdx = b.jdqdx(i, j, k, 3);

        dudy = b.jdqdy(i, j, k, 1);
        dvdy = b.jdqdy(i, j, k, 2);
        dwdy = b.jdqdy(i, j, k, 3);

        dudz = b.jdqdz(i, j, k, 1);
        dvdz = b.jdqdz(i, j, k, 2);
        dwdz = b.jdqdz(i, j, k, 3);

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
        dTdx = b.jdqdx(i, j, k, 4);
        dTdy = b.jdqdy(i, j, k, 4);
        dTdz = b.jdqdz(i, j, k, 4);

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
        double gradYk, Dij, Dcorr = 0.0;
        double gradYns = 0.0;
        double rho = 0.5 * (b.Q(i, j, k, 0) + b.Q(i, j - 1, k, 0));
        // Compute the species flux and correction term \sum(k=1,ns) Dij*gradYk
        for (int n = 0; n < b.ne - 5; n++) {
          Dij = 0.5 * (b.qt(i, j, k, 2 + n) + b.qt(i, j - 1, k, 2 + n));
          dYdx = b.jdqdx(i, j, k, 5 + n);
          dYdy = b.jdqdy(i, j, k, 5 + n);
          dYdz = b.jdqdz(i, j, k, 5 + n);
          gradYk = (dYdx * b.jsx(i, j, k) + dYdy * b.jsy(i, j, k) +
                    dYdz * b.jsz(i, j, k));
          gradYns -= gradYk;
          Dcorr += Dij * gradYk;
          b.jF(i, j, k, 5 + n) = -rho * Dij * gradYk;
        }
        // Apply n=ns species to correction
        Dij = 0.5 *
              (b.qt(i, j, k, 2 + b.ne - 5) + b.qt(i, j - 1, k, 2 + b.ne - 5));
        Dcorr += Dij * gradYns;

        // Apply correction and species thermal flux
        double Yk, hk;
        double Yns = 1.0;
        for (int n = 0; n < b.ne - 5; n++) {
          Yk = 0.5 * (b.q(i, j, k, 5 + n) + b.q(i, j - 1, k, 5 + n));
          Yns -= Yk;
          b.jF(i, j, k, 5 + n) += Yk * rho * Dcorr;
          // Species thermal diffusion
          hk = 0.5 * (b.qh(i, j, k, 5 + n) + b.qh(i, j - 1, k, 5 + n));
          b.jF(i, j, k, 4) += b.jF(i, j, k, 5 + n) * hk;
        }
        // Apply the n=ns species to thermal diffusion
        Yns = fmax(Yns, 0.0);
        hk = 0.5 * (b.qh(i, j, k, b.ne) + b.qh(i, j - 1, k, b.ne));
        b.jF(i, j, k, 4) += (-rho * Dij * gradYns + Yns * rho * Dcorr) * hk;
      });

  //-------------------------------------------------------------------------------------------|
  // k flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_k({b.ng, b.ng, b.ng},
                   {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng});
  Kokkos::parallel_for(
      "k face visc fluxes", range_k,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double dudx, dudy, dudz;
        double dvdx, dvdy, dvdz;
        double dwdx, dwdy, dwdz;

        double txx, txy, txz;
        double tyx, tyy, tyz;
        double tzx, tzy, tzz;

        double dTdx, dTdy, dTdz;
        double uf, vf, wf;
        double q;

        double dYdx, dYdy, dYdz;

        const double c23 = 2.0 / 3.0;

        double mu = 0.5 * (b.qt(i, j, k, 0) + b.qt(i, j, k - 1, 0));
        double kappa = 0.5 * (b.qt(i, j, k, 1) + b.qt(i, j, k - 1, 1));

        // continuity
        b.kF(i, j, k, 0) = 0.0;

        // Spatial derivative on face
        dudx = b.kdqdx(i, j, k, 1);
        dvdx = b.kdqdx(i, j, k, 2);
        dwdx = b.kdqdx(i, j, k, 3);

        dudy = b.kdqdy(i, j, k, 1);
        dvdy = b.kdqdy(i, j, k, 2);
        dwdy = b.kdqdy(i, j, k, 3);

        dudz = b.kdqdz(i, j, k, 1);
        dvdz = b.kdqdz(i, j, k, 2);
        dwdz = b.kdqdz(i, j, k, 3);

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
        dTdx = b.kdqdx(i, j, k, 4);
        dTdy = b.kdqdy(i, j, k, 4);
        dTdz = b.kdqdz(i, j, k, 4);

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
        double gradYk, Dij, Dcorr = 0.0;
        double gradYns = 0.0;
        double rho = 0.5 * (b.Q(i, j, k, 0) + b.Q(i, j, k - 1, 0));
        // Compute the species flux and correction term \sum(k=1,ns) Dij*gradYk
        for (int n = 0; n < b.ne - 5; n++) {
          Dij = 0.5 * (b.qt(i, j, k, 2 + n) + b.qt(i, j, k - 1, 2 + n));
          dYdx = b.kdqdx(i, j, k, 5 + n);
          dYdy = b.kdqdy(i, j, k, 5 + n);
          dYdz = b.kdqdz(i, j, k, 5 + n);
          gradYk = (dYdx * b.ksx(i, j, k) + dYdy * b.ksy(i, j, k) +
                    dYdz * b.ksz(i, j, k));
          gradYns -= gradYk;
          Dcorr += Dij * gradYk;
          b.kF(i, j, k, 5 + n) = -rho * Dij * gradYk;
        }
        // Apply n=ns species to correction
        Dij = 0.5 *
              (b.qt(i, j, k, 2 + b.ne - 5) + b.qt(i, j, k - 1, 2 + b.ne - 5));
        Dcorr += Dij * gradYns;

        // Apply correction and species thermal flux
        double Yk, hk;
        double Yns = 1.0;
        for (int n = 0; n < b.ne - 5; n++) {
          Yk = 0.5 * (b.q(i, j, k, 5 + n) + b.q(i, j, k - 1, 5 + n));
          Yns -= Yk;
          b.kF(i, j, k, 5 + n) += Yk * rho * Dcorr;
          // Species thermal diffusion
          hk = 0.5 * (b.qh(i, j, k, 5 + n) + b.qh(i, j, k - 1, 5 + n));
          b.kF(i, j, k, 4) += b.kF(i, j, k, 5 + n) * hk;
        }
        // Apply the n=ns species to thermal diffusion
        Yns = fmax(Yns, 0.0);
        hk = 0.5 * (b.qh(i, j, k, b.ne) + b.qh(i, j, k - 1, b.ne));
        b.kF(i, j, k, 4) += (-rho * Dij * gradYns + Yns * rho * Dcorr) * hk;
      });
}

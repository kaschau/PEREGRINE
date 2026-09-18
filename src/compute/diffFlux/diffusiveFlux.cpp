#include "faces.hpp"
#include "thermo/eos.hpp"

// This should basically never be used. It always worse than alpha damping.

PG_RANGE(cellFaces)
struct diffusiveFlux {
  cellCenterL QL, gradsL, qL, qhL, qtL;
  cellCenterR QR, gradsR, qR, qhR, qtR;
  cellFaceInOut F;
  cellFaceIn A;
  static constexpr fpdtype bulkVisc = 0.0;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    fpdtype mu = 0.5 * (qtR(0) + qtL(0));
    fpdtype kappa = 0.5 * (qtR(1) + qtL(1));
    fpdtype lambda = bulkVisc - 2.0 / 3.0 * mu;

    // no mass diffuses, so the continuity flux is left alone

    // Derivatives on face
    fpdtype dudx = 0.5 * (gradsR(0, 0) + gradsL(0, 0));
    fpdtype dvdx = 0.5 * (gradsR(1, 0) + gradsL(1, 0));
    fpdtype dwdx = 0.5 * (gradsR(2, 0) + gradsL(2, 0));

    fpdtype dudy = 0.5 * (gradsR(0, 1) + gradsL(0, 1));
    fpdtype dvdy = 0.5 * (gradsR(1, 1) + gradsL(1, 1));
    fpdtype dwdy = 0.5 * (gradsR(2, 1) + gradsL(2, 1));

    fpdtype dudz = 0.5 * (gradsR(0, 2) + gradsL(0, 2));
    fpdtype dvdz = 0.5 * (gradsR(1, 2) + gradsL(1, 2));
    fpdtype dwdz = 0.5 * (gradsR(2, 2) + gradsL(2, 2));

    fpdtype div = dudx + dvdy + dwdz;

    // x momentum
    fpdtype txx = -2.0 * mu * dudx - lambda * div;
    fpdtype txy = -mu * (dvdx + dudy);
    fpdtype txz = -mu * (dwdx + dudz);

    F(1) += txx * A(0) + txy * A(1) + txz * A(2);

    // y momentum
    fpdtype &tyx = txy;
    fpdtype tyy = -2.0 * mu * dvdy - lambda * div;
    fpdtype tyz = -mu * (dwdy + dvdz);

    F(2) += tyx * A(0) + tyy * A(1) + tyz * A(2);

    // z momentum
    fpdtype &tzx = txz;
    fpdtype &tzy = tyz;
    fpdtype tzz = -2.0 * mu * dwdz - lambda * div;

    F(3) += tzx * A(0) + tzy * A(1) + tzz * A(2);

    // energy
    //   heat conduction
    fpdtype dTdx = 0.5 * (gradsR(3, 0) + gradsL(3, 0));
    fpdtype dTdy = 0.5 * (gradsR(3, 1) + gradsL(3, 1));
    fpdtype dTdz = 0.5 * (gradsR(3, 2) + gradsL(3, 2));

    fpdtype heatFlux = -kappa * (dTdx * A(0) + dTdy * A(1) + dTdz * A(2));

    // flow work
    // Compute face normal volume flux vector
    // the face velocity, each side's off its conserved state
    const fpdtype rhoinvL = 1.0 / QL(0), rhoinvR = 1.0 / QR(0);
    fpdtype uf = 0.5 * (QR(1) * rhoinvR + QL(1) * rhoinvL);
    fpdtype vf = 0.5 * (QR(2) * rhoinvR + QL(2) * rhoinvL);
    fpdtype wf = 0.5 * (QR(3) * rhoinvR + QL(3) * rhoinvL);

    F(4) += -(uf * txx + vf * txy + wf * txz) * A(0) -
            (uf * tyx + vf * tyy + wf * tyz) * A(1) -
            (uf * tzx + vf * tzy + wf * tzz) * A(2) + heatFlux;

    // Species
    // the species enthalpies from each side's eos
    const auto hL = eos::enthalpies(qL(1), qhL);
    const auto hR = eos::enthalpies(qR(1), qhR);
    fpdtype Dk, Vc = 0.0;
    fpdtype gradYns = 0.0;
    fpdtype rho = 0.5 * (QR(0) + QL(0));
    // Compute the species flux and correction term \sum(k=1,ns)
    // Dk*gradYk
    for (int n = 0; n < ne - 5; n++) {
      Dk = 0.5 * (qtR(2 + n) + qtL(2 + n));
      fpdtype dYdx = 0.5 * (gradsR(4 + n, 0) + gradsL(4 + n, 0));
      fpdtype dYdy = 0.5 * (gradsR(4 + n, 1) + gradsL(4 + n, 1));
      fpdtype dYdz = 0.5 * (gradsR(4 + n, 2) + gradsL(4 + n, 2));

      fpdtype gradYk = (dYdx * A(0) + dYdy * A(1) + dYdz * A(2));
      gradYns -= gradYk;
      Vc += Dk * gradYk;
      // the species flux before its correction, and its enthalpy with
      // it
      fpdtype Jk = -rho * Dk * gradYk;
      fpdtype hk = 0.5 * (hR(n) + hL(n));
      F(5 + n) += Jk;
      F(4) += Jk * hk;
    }
    // Apply n=ns species to correction
    Dk = 0.5 * (qtR(2 + ne - 5) + qtL(2 + ne - 5));
    Vc += Dk * gradYns;

    // Apply correction and species thermal flux
    fpdtype Yk, hk;
    fpdtype Yns = 1.0;
    for (int n = 0; n < ne - 5; n++) {
      Yk = 0.5 * (QR(5 + n) * rhoinvR + QL(5 + n) * rhoinvL);
      Yns -= Yk;
      // the correction, and its enthalpy with it
      fpdtype corr = Yk * rho * Vc;
      hk = 0.5 * (hR(n) + hL(n));
      F(5 + n) += corr;
      F(4) += corr * hk;
    }
    // Apply the n=ns species to thermal diffusion
    Yns = fmax(Yns, 0.0);
    hk = 0.5 * (hR(ns - 1) + hL(ns - 1));
    F(4) += (-rho * Dk * gradYns + Yns * rho * Vc) * hk;
  }
};

PG_ABI void pgDiffusiveFlux(const diffusiveFlux &k, const pgTiling &t) {
  forCells("i face visc fluxes", t, k);
}

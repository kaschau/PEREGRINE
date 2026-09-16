#include "faces.hpp"
#include "thermo/eos.hpp"

// References
//
// Robust numerical fluxes for unrealizable states
//
// Hiroaki Nishikawa
// Journal of Computational Physics
// 408 (2020)

// grads holds the velocity in slots 0 .. 2, T in 3 and Y(n) in 4 + n; the
// velocity and Y at each side come off its conserved state, the species
// enthalpies from the eos at its temperature.
PG_RANGE(cellFaces)
struct alphaDampingFlux {
  cellCenterL QL, cellsL, gradsL, qL, qhL, qtL;
  cellCenterR QR, cellsR, gradsR, qR, qhR, qtR;
  cellFaceInOut F;
  cellFaceIn A, Faces;
  static constexpr double bulkVisc = 0.0;
  static constexpr double alpha = 1.0;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    double S, nx, ny, nz;
    faceNormal(A(0), A(1), A(2), S, nx, ny, nz);

    double mu = 0.5 * (qtR(0) + qtL(0));
    double kappa = 0.5 * (qtR(1) + qtL(1));
    double lambda = bulkVisc - 2.0 / 3.0 * mu;

    // each side's velocity, off its conserved state
    const double rhoinvL = 1.0 / QL(0), rhoinvR = 1.0 / QR(0);
    const double uL = QL(1) * rhoinvL, vL = QL(2) * rhoinvL,
                 wL = QL(3) * rhoinvL;
    const double uR = QR(1) * rhoinvR, vR = QR(2) * rhoinvR,
                 wR = QR(3) * rhoinvR;

    // Compute face normal gradient with alpha damping
    const double ex = cellsR(0) - cellsL(0), ey = cellsR(1) - cellsL(1),
                 ez = cellsR(2) - cellsL(2);
    const double en0 = ex * nx, en1 = ey * ny, en2 = ez * nz;
    const double dampInv = alpha / sqrt(en0 * en0 + en1 * en1 + en2 * en2);
    const double dampx = dampInv * nx, dampy = dampInv * ny,
                 dampz = dampInv * nz;
    const double xLx = Faces(0) - cellsL(0), xLy = Faces(1) - cellsL(1),
                 xLz = Faces(2) - cellsL(2);
    const double xRx = Faces(0) - cellsR(0), xRy = Faces(1) - cellsR(1),
                 xRz = Faces(2) - cellsR(2);

    // the face gradient of slot l of grads: the sides' average, damped
    // toward the difference of the values each side extrapolates to the face
    const auto faceGrad = [&](const int l, const double phiL, const double phiR,
                              double &gx, double &gy, double &gz) {
      const double gLx = gradsL(l, 0), gLy = gradsL(l, 1), gLz = gradsL(l, 2);
      const double gRx = gradsR(l, 0), gRy = gradsR(l, 1), gRz = gradsR(l, 2);
      const double wl = phiL + gLx * xLx + gLy * xLy + gLz * xLz;
      const double wr = phiR + gRx * xRx + gRy * xRy + gRz * xRz;
      const double jump = wr - wl;
      gx = 0.5 * (gRx + gLx) + dampx * jump;
      gy = 0.5 * (gRy + gLy) + dampy * jump;
      gz = 0.5 * (gRz + gLz) + dampz * jump;
    };

    double dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz;
    faceGrad(0, uL, uR, dudx, dudy, dudz);
    faceGrad(1, vL, vR, dvdx, dvdy, dvdz);
    faceGrad(2, wL, wR, dwdx, dwdy, dwdz);

    double div = dudx + dvdy + dwdz;

    // x momentum
    double txx = -2.0 * mu * dudx - lambda * div;
    double txy = -mu * (dvdx + dudy);
    double txz = -mu * (dwdx + dudz);

    F(1) += txx * A(0) + txy * A(1) + txz * A(2);

    // y momentum
    double &tyx = txy;
    double tyy = -2.0 * mu * dvdy - lambda * div;
    double tyz = -mu * (dwdy + dvdz);

    F(2) += tyx * A(0) + tyy * A(1) + tyz * A(2);

    // z momentum
    double &tzx = txz;
    double &tzy = tyz;
    double tzz = -2.0 * mu * dwdz - lambda * div;

    F(3) += tzx * A(0) + tzy * A(1) + tzz * A(2);

    // energy
    double dTx, dTy, dTz;
    faceGrad(3, qL(1), qR(1), dTx, dTy, dTz);
    double heatFlux = -kappa * (dTx * A(0) + dTy * A(1) + dTz * A(2));

    double uf = 0.5 * (uR + uL);
    double vf = 0.5 * (vR + vL);
    double wf = 0.5 * (wR + wL);

    const double work = -(uf * txx + vf * txy + wf * txz) * A(0) -
                        (uf * tyx + vf * tyy + wf * tyz) * A(1) -
                        (uf * tzx + vf * tzy + wf * tzz) * A(2) + heatFlux;

    // species, with the correction that keeps the diffusion fluxes summing
    // to zero; the enthalpies from each side's eos. The energy flux gathers
    // in a register: F(4) is written once. The correction needs the sum
    // over every species first, so each species' flux waits in F between
    // the two passes (measured on an MI100 at 53 species, the read-back is
    // 29% of the kernel; registers would hold it to ~16 species)
    const auto hL = eos::enthalpies(qL(1), qhL);
    const auto hR = eos::enthalpies(qR(1), qhR);
    double Dk, Vc = 0.0, gradYns = 0.0, sumYh = 0.0, Yns = 1.0, f4 = work;
    double rho = 0.5 * (QR(0) + QL(0));
    for (int n = 0; n < ne - 5; n++) {
      Dk = 0.5 * (qtR(2 + n) + qtL(2 + n));
      const double YkL = QL(5 + n) * rhoinvL, YkR = QR(5 + n) * rhoinvR;
      double dYx, dYy, dYz;
      faceGrad(4 + n, YkL, YkR, dYx, dYy, dYz);
      double gradYk = (dYx * A(0) + dYy * A(1) + dYz * A(2));
      gradYns -= gradYk;
      Vc += Dk * gradYk;
      double Jk = -rho * Dk * gradYk;
      double hk = 0.5 * (hR(n) + hL(n));
      const double Yk = 0.5 * (YkR + YkL);
      Yns -= Yk;
      sumYh += Yk * hk;
      F(5 + n) = F(5 + n) + Jk;
      f4 += Jk * hk;
    }
    Dk = 0.5 * (qtR(2 + ne - 5) + qtL(2 + ne - 5));
    Vc += Dk * gradYns;

    // the correction: Yk rho Vc to each species, its enthalpy-weighted sum
    // to the energy
    const double corr = rho * Vc;
    for (int n = 0; n < ne - 5; n++) {
      F(5 + n) += 0.5 * (QR(5 + n) * rhoinvR + QL(5 + n) * rhoinvL) * corr;
    }
    f4 += corr * sumYh;
    const double hns = 0.5 * (hR(ns - 1) + hL(ns - 1));
    f4 += (-rho * Dk * gradYns + Yns * corr) * hns;
    F(4) += f4;
  }
};

PG_ABI void pgAlphaDampingFlux(const alphaDampingFlux &k, const pgTiling &t) {
  forCells("face visc fluxes", t, k);
}

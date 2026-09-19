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
  cellStradVecIn Q, cells, q, qh, qt;
  cellStradMatIn grads;
  faceVecInOut F;
  faceVecIn A, Faces;
  static constexpr fpdtype bulkVisc = 0.0;
  static constexpr fpdtype alpha = 1.0;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    fpdtype S, nx, ny, nz;
    faceNormal(A(0), A(1), A(2), S, nx, ny, nz);

    fpdtype mu = 0.5 * (qt.R(0) + qt.L(0));
    fpdtype kappa = 0.5 * (qt.R(1) + qt.L(1));
    fpdtype lambda = bulkVisc - 2.0 / 3.0 * mu;

    // each side's velocity, off its conserved state
    const fpdtype rhoinvL = 1.0 / Q.L(0), rhoinvR = 1.0 / Q.R(0);
    const fpdtype uL = Q.L(1) * rhoinvL, vL = Q.L(2) * rhoinvL,
                  wL = Q.L(3) * rhoinvL;
    const fpdtype uR = Q.R(1) * rhoinvR, vR = Q.R(2) * rhoinvR,
                  wR = Q.R(3) * rhoinvR;

    // Compute face normal gradient with alpha damping
    const fpdtype ex = cells.R(0) - cells.L(0), ey = cells.R(1) - cells.L(1),
                  ez = cells.R(2) - cells.L(2);
    const fpdtype en0 = ex * nx, en1 = ey * ny, en2 = ez * nz;
    const fpdtype dampInv = alpha / sqrt(en0 * en0 + en1 * en1 + en2 * en2);
    const fpdtype dampx = dampInv * nx, dampy = dampInv * ny,
                  dampz = dampInv * nz;
    const fpdtype xLx = Faces(0) - cells.L(0), xLy = Faces(1) - cells.L(1),
                  xLz = Faces(2) - cells.L(2);
    const fpdtype xRx = Faces(0) - cells.R(0), xRy = Faces(1) - cells.R(1),
                  xRz = Faces(2) - cells.R(2);

    // the face gradient of slot l of grads: the sides' average, damped
    // toward the difference of the values each side extrapolates to the face
    const auto faceGrad = [&](const int l, const fpdtype phiL,
                              const fpdtype phiR, fpdtype &gx, fpdtype &gy,
                              fpdtype &gz) {
      const fpdtype gLx = grads.L(l, 0), gLy = grads.L(l, 1),
                    gLz = grads.L(l, 2);
      const fpdtype gRx = grads.R(l, 0), gRy = grads.R(l, 1),
                    gRz = grads.R(l, 2);
      const fpdtype wl = phiL + gLx * xLx + gLy * xLy + gLz * xLz;
      const fpdtype wr = phiR + gRx * xRx + gRy * xRy + gRz * xRz;
      const fpdtype jump = wr - wl;
      gx = 0.5 * (gRx + gLx) + dampx * jump;
      gy = 0.5 * (gRy + gLy) + dampy * jump;
      gz = 0.5 * (gRz + gLz) + dampz * jump;
    };

    fpdtype dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz;
    faceGrad(0, uL, uR, dudx, dudy, dudz);
    faceGrad(1, vL, vR, dvdx, dvdy, dvdz);
    faceGrad(2, wL, wR, dwdx, dwdy, dwdz);

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
    fpdtype dTx, dTy, dTz;
    faceGrad(3, q.L(1), q.R(1), dTx, dTy, dTz);
    fpdtype heatFlux = -kappa * (dTx * A(0) + dTy * A(1) + dTz * A(2));

    fpdtype uf = 0.5 * (uR + uL);
    fpdtype vf = 0.5 * (vR + vL);
    fpdtype wf = 0.5 * (wR + wL);

    const fpdtype work = -(uf * txx + vf * txy + wf * txz) * A(0) -
                         (uf * tyx + vf * tyy + wf * tyz) * A(1) -
                         (uf * tzx + vf * tzy + wf * tzz) * A(2) + heatFlux;

    // species, with the correction that keeps the diffusion fluxes summing
    // to zero; the enthalpies from each side's eos. The energy flux gathers
    // in a register: F(4) is written once. The correction needs the sum
    // over every species first, so each species' flux waits in F between
    // the two passes (measured on an MI100 at 53 species, the read-back is
    // 29% of the kernel; registers would hold it to ~16 species)
    const auto hL = eos::enthalpies(q.L(1), qh.L());
    const auto hR = eos::enthalpies(q.R(1), qh.R());
    fpdtype Dk, Vc = 0.0, gradYns = 0.0, sumYh = 0.0, Yns = 1.0, f4 = work;
    fpdtype rho = 0.5 * (Q.R(0) + Q.L(0));
    for (int n = 0; n < ne - 5; n++) {
      Dk = 0.5 * (qt.R(2 + n) + qt.L(2 + n));
      const fpdtype YkL = Q.L(5 + n) * rhoinvL, YkR = Q.R(5 + n) * rhoinvR;
      fpdtype dYx, dYy, dYz;
      faceGrad(4 + n, YkL, YkR, dYx, dYy, dYz);
      fpdtype gradYk = (dYx * A(0) + dYy * A(1) + dYz * A(2));
      gradYns -= gradYk;
      Vc += Dk * gradYk;
      fpdtype Jk = -rho * Dk * gradYk;
      fpdtype hk = 0.5 * (hR(n) + hL(n));
      const fpdtype Yk = 0.5 * (YkR + YkL);
      Yns -= Yk;
      sumYh += Yk * hk;
      F(5 + n) = F(5 + n) + Jk;
      f4 += Jk * hk;
    }
    Dk = 0.5 * (qt.R(2 + ne - 5) + qt.L(2 + ne - 5));
    Vc += Dk * gradYns;

    // the correction: Yk rho Vc to each species, its enthalpy-weighted sum
    // to the energy
    const fpdtype corr = rho * Vc;
    for (int n = 0; n < ne - 5; n++) {
      F(5 + n) += 0.5 * (Q.R(5 + n) * rhoinvR + Q.L(5 + n) * rhoinvL) * corr;
    }
    f4 += corr * sumYh;
    const fpdtype hns = 0.5 * (hR(ns - 1) + hL(ns - 1));
    f4 += (-rho * Dk * gradYns + Yns * corr) * hns;
    F(4) += f4;
  }
};

PG_ABI void pgAlphaDampingFlux(const alphaDampingFlux &k, const pgTiling &t) {
  forCells("face visc fluxes", t, k);
}

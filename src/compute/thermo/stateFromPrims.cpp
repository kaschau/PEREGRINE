#include "eos.hpp"
#include "kernel.hpp"

// The state of every cell from a primitive vector p, u, v, w, T, Y(0 .. ns
// - 2), which a case starts from: the mass fractions clipped and
// renormalized, then the case's eos for the rest.
PG_RANGE(cellCenters)
struct stateFromPrims {
  cellCenterIn prims;
  cellCenterOut Q, q, qh;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const double p = prims(0);
    const double u = prims(1);
    const double v = prims(2);
    const double w = prims(3);
    const double T = prims(4);

    double Y[ns];
    Y[ns - 1] = 1.0;
    double testSum = 0.0;
    for (int n = 0; n < ns - 1; n++) {
      Y[n] = fmax(fmin(prims(5 + n), 1.0), 0.0);
      Y[ns - 1] -= Y[n];
      testSum += Y[n];
    }
    if (testSum > 1.0) {
      Y[ns - 1] = 0.0;
      const double sumInv = 1.0 / testSum;
      for (int n = 0; n < ns - 1; n++) {
        Y[n] *= sumInv;
      }
    }

    const auto Yof = [&](const int n) { return Y[n]; };
    const auto s = eos::fromPrims(p, T, Yof, keepHi(qh));
    writeState(s, u, v, w, Yof, Q, q, qh);
  }
};

PG_ABI void pgStateFromPrims(const stateFromPrims &k, const pgTiling &t) {
  forCells("state from primitives", t, k);
}

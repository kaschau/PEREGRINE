#include "block_.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>

//-------------------------------------------------------------------------------------------|
// Q = wQ0 * Q0 + wQ * Q + wdQ * dt * dQ
//
// Every strong stability preserving stage takes this form. Whether the stage
// stores the state the step began from, and whether it reads it back, are
// template parameters so the kernel itself carries no branch.
//-------------------------------------------------------------------------------------------|
template <bool storeQ0, bool readQ0>
static void stage(block_ &b, const double &dt, const double &wQ0,
                  const double &wQ, const double &wdQ) {
  MDRange4 range_cc({b.ng, b.ng, b.ng, 0},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
  Kokkos::parallel_for(
      "stage update", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        if constexpr (storeQ0) {
          b.Q0(i, j, k, l) = b.Q(i, j, k, l);
        }
        if constexpr (readQ0) {
          b.Q(i, j, k, l) = wQ0 * b.Q0(i, j, k, l) + wQ * b.Q(i, j, k, l) +
                            wdQ * dt * b.dQ(i, j, k, l);
        } else {
          // a stage that does not look back does not pay to read Q0
          b.Q(i, j, k, l) = wQ * b.Q(i, j, k, l) + wdQ * dt * b.dQ(i, j, k, l);
        }
      });
}

void applyStage(block_ &b, const double &dt, const double &wQ0,
                const double &wQ, const double &wdQ, const bool &storeQ0) {
  const bool readQ0 = (wQ0 != 0.0);
  if (storeQ0 && readQ0) {
    stage<true, true>(b, dt, wQ0, wQ, wdQ);
  } else if (storeQ0) {
    stage<true, false>(b, dt, wQ0, wQ, wdQ);
  } else if (readQ0) {
    stage<false, true>(b, dt, wQ0, wQ, wdQ);
  } else {
    stage<false, false>(b, dt, wQ0, wQ, wdQ);
  }
}

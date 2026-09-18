// The four cells about the face as they are: piecewise constant's two
// states, and the cells beyond them for a formula that reaches that far.
#ifndef __reconstructFourCells_H__
#define __reconstructFourCells_H__

#include "advFlux/reconstruct/piecewiseConstant.hpp"
#include "faces.hpp"

struct fourCells : piecewiseConstant {
  using base = piecewiseConstant;
  cellCenterLL QLL, qLL, qhLL;
  cellCenterRR QRR, qRR, qhRR;
  template <class P> KOKKOS_INLINE_FUNCTION void pin(const P &at) {
    pinKernel(static_cast<base &>(*this), at);
    pinEach(at, QLL, qLL, qhLL, QRR, qRR, qhRR);
  }
};

#endif

// KEPaEC to fourth order: the two-point flux over the pairs two cells
// either side of the face, in the standard high order form (Pirozzoli
// 2010) -- 4/3 of the near pair less 1/6 of each pair a cell apart -- which
// keeps the two-point flux's properties. The four cells as they are:
// PG_STENCIL(2).
#ifndef __formulaFourthOrderKEPaEC_H__
#define __formulaFourthOrderKEPaEC_H__

#include "advFlux/fluxOut.hpp"
#include "advFlux/formula/KEPaEC.hpp"
#include "advFlux/reconstruct/piecewiseConstant.hpp"

PG_STENCIL(2);

struct fourthOrderKEPaEC {
  template <class Recon, class Out>
  static KOKKOS_INLINE_FUNCTION void flux(const Recon &r, const faceVecIn &A,
                                          const Out &F) {
    static_assert(std::is_base_of_v<piecewiseConstant, Recon>,
                  "fourthOrderKEPaEC takes the four cells about the face");
    KEPaEC::twoPoint(r.Q.L(), r.q.L(), r.qh.L(), r.Q.R(), r.q.R(), r.qh.R(), A,
                     weighted<Out>{F, 4.0 / 3.0});
    KEPaEC::twoPoint(r.Q.LL(), r.q.LL(), r.qh.LL(), r.Q.R(), r.q.R(), r.qh.R(),
                     A, added<Out>{F, -1.0 / 6.0});
    KEPaEC::twoPoint(r.Q.L(), r.q.L(), r.qh.L(), r.Q.RR(), r.q.RR(), r.qh.RR(),
                     A, added<Out>{F, -1.0 / 6.0});
  }
};

#endif

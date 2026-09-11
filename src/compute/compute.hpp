#ifndef __compute_H__
#define __compute_H__

#include "block_.hpp"
#include "face_.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"
#include "vector"
#include <Kokkos_Core.hpp>
#include <string>

///////////////////////////////////////////////////////////
////////////////// c++ Only Functions /////////////////////
///////////////////////////////////////////////////////////

// ./range_map.cpp
MDRange3 getRange3(const block_ &b, const int &nface, const int &indxI = 0,
                   const int &indxJ = 0, const int &indxK = 0);
twoDsubview getFaceSlice(const threeDview &view, const int &nface,
                         const int &slice);
threeDsubview getFaceSlice(const fourDview &view, const int &nface,
                           const int &slice);
fourDsubview getFaceSlice(const fiveDview &view, const int &nface,
                          const int &slice);
// the area vector of a face, from whichever of the i, j, k face arrays it
// belongs to
threeDsubview getFaceAreaVectors(const block_ &b, const int &nface,
                                 const int &slice);
void getFaceSliceIdxs(int &firstHaloIdx, int &s1, int &s2, int &plus,
                      const int &ni, const int &nj, const int &nk,
                      const int &ng, const int &nface);

// a face's area and unit normal, from its area vector
KOKKOS_INLINE_FUNCTION
void faceNormal(const double &sx, const double &sy, const double &sz, double &S,
                double &nx, double &ny, double &nz) {
  // a degenerate face is floored, we divide by this
  S = fmax(sqrt(sx * sx + sy * sy + sz * sz), 1e-16);
  nx = sx / S;
  ny = sy / S;
  nz = sz / S;
}

#endif

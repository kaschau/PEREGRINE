#ifndef __compute_H__
#define __compute_H__

#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "face_.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"
#include "vector"
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
void getFaceSliceIdxs(int &firstHaloIdx, int &s1, int &s2, int &plus,
                      const int &ni, const int &nj, const int &nk,
                      const int &ng, const int &nface);

#endif

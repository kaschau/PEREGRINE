// A kernel on the cell faces of one direction, which is what every flux
// is: compiled once per direction, the jit says which, and the index
// offsets fold into the addressing. kernel.hpp with the direction on it.
#ifndef __faces_H__
#define __faces_H__

#include "kernel.hpp"

#ifndef PG_DIRECTION
#error                                                                         \
    "a flux kernel is compiled for one direction: -DPG_DIRECTION from the jit"
#endif
constexpr int iMod = PG_DIRECTION == 0, jMod = PG_DIRECTION == 1,
              kMod = PG_DIRECTION == 2;
// the face normal as a step between cells
constexpr offset N{iMod, jMod, kMod};

// a cell-center column seen from the face the thread traverses: L is the
// cell to its left, R the cell to its right, which the face is indexed
// like; LL and RR one further
using cellCenterL = column<const double, -N>;
using cellCenterR = cellCenterIn;
using cellCenterLL = column<const double, -2 * N>;
using cellCenterRR = column<const double, N>;

#endif

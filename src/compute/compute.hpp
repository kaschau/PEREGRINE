#ifndef __compute_H__
#define __compute_H__

#include "kokkos_types.hpp"
#include "Block.hpp"

// Define compute functions
void add3D(Block b, double n);

threeDview gen3Dview(std::string name, int ni, int nj, int nk);
fourDview  gen4Dview(std::string name, int ni, int nj, int nk, int nl);

void finalize_kokkos();

#endif

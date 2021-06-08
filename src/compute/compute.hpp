#ifndef __compute_H__
#define __compute_H__

#include "kokkos_types.hpp"
#include "block_.hpp"

// Temporary creation functions
threeDview gen3Dview(std::string name, int ni, int nj, int nk);
fourDview  gen4Dview(std::string name, int ni, int nj, int nk, int nl);

void finalize_kokkos();

///////////////////////////////////////////////////////////
////////////////// Compute Functions //////////////////////
///////////////////////////////////////////////////////////

// METRICS
void metrics(block_ b);


//temp
void add3D(block_ b, double n);
#endif

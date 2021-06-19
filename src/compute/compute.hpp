#ifndef __compute_H__
#define __compute_H__

#include "kokkos_types.hpp"
#include "block_.hpp"

// Temporary creation functions
threeDview gen3Dview(std::string name, int ni, int nj, int nk);
fourDview  gen4Dview(std::string name, int ni, int nj, int nk, int nl);


///////////////////////////////////////////////////////////
////////////////// Compute Functions //////////////////////
///////////////////////////////////////////////////////////

// ./grid
//    |------> Metrics
void metrics(block_ b);

// ./flux
//    |------> Advective
void advective(block_ b);
void apply_flux(block_ b);

// ./estr
//    |------> total_energy
void total_energy(block_ b);




#endif

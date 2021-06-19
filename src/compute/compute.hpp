#ifndef __compute_H__
#define __compute_H__

#include "kokkos_types.hpp"
#include "block_.hpp"
#include <vector>

///////////////////////////////////////////////////////////
////////////////// Compute Functions //////////////////////
///////////////////////////////////////////////////////////

// ./grid
//    |------> Metrics
void metrics(std::vector<block_> mb);

// ./flux
//    |------> Advective
void advective(std::vector<block_> mb);
void apply_flux(std::vector<block_> mb);

// ./estr
//    |------> total_energy
void total_energy(std::vector<block_> mb);




#endif

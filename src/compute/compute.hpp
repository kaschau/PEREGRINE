#ifndef __compute_H__
#define __compute_H__

#include "kokkos_types.hpp"
#include "block_.hpp"
#include "thermdat_.hpp"
#include <vector>
#include <string>

///////////////////////////////////////////////////////////
////////////////// Compute Functions //////////////////////
///////////////////////////////////////////////////////////
MDRange3 get_range3(block_ b, int face);

// ./grid
//    |------> Metrics
void metrics(std::vector<block_> mb);

// ./flux
//    |------> dQzero
void dQzero(std::vector<block_> mb);
//    |------> dqdxyz
void dqdxyz(std::vector<block_> mb);
//    |------> Advective
void advective(std::vector<block_> mb, thermdat_ th);
//    |------> Diffusive
void diffusive(std::vector<block_> mb, thermdat_ th);

// ./thermo
//    |------> cpg
void cpg(block_ b, thermdat_ th, int face, std::string given);
//    |------> tpg
void tpg(block_ b, thermdat_ th, int face, std::string given);

#endif

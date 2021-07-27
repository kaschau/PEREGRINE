#ifndef __compute_H__
#define __compute_H__

#include "kokkos_types.hpp"
#include "block_.hpp"
#include <vector>
#include <string>

///////////////////////////////////////////////////////////
////////////////// Compute Functions //////////////////////
///////////////////////////////////////////////////////////
MDRange3 get_range3(block_ b, std::string face);

// ./grid
//    |------> Metrics
void metrics(std::vector<block_> mb);

// ./flux
//    |------> Advective
void advective(std::vector<block_> mb);

// ./EOS
//    |------> EOS_ideal
void EOS_ideal(block_ b, std::string face, std::string given);
//    |------> calEOS_perfect
void calEOS_perfect(block_ b, std::string face, std::string given);

// ./consistify
//    |------> momentum
void momentum(block_ b, std::string face, std::string given);

#endif

#ifndef __timeIntegration_H__
#define __timeIntegration_H__

#include "block_.hpp"
#include "thtrdat_.hpp"
#include "vector"

// ./timeIntegration
//    |------> dualTime.cpp
void dQdt(block_ b, const double dt);
void localDtau(block_ b, const bool viscous);
void DTrk3s1(block_ b);
void DTrk3s2(block_ b);
void DTrk3s3(block_ b);
std::array<std::vector<double>, 3> residual(std::vector<block_> mb);
void invertDQ(block_ b, const double dt, const thtrdat_ th, const bool viscous);
//    |------> maccormack.cpp
void corrector(block_ b, const double dt);
//    |------> rk2Stages.cpp
void rk2s1(block_ b, const double dt);
void rk2s2(block_ b, const double dt);
//    |------> rk3Stages.cpp
void rk3s1(block_ b, const double dt);
void rk3s2(block_ b, const double dt);
void rk3s3(block_ b, const double dt);
//    |------> rk4Stages.cpp
void rk4s1(block_ b, const double dt);
void rk4s2(block_ b, const double dt);
void rk4s3(block_ b, const double dt);
void rk4s4(block_ b, const double dt);

#endif

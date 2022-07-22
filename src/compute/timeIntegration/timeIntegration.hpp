#ifndef __timeIntegration_H__
#define __timeIntegration_H__

#include "block_.hpp"

// ./timeIntegration
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

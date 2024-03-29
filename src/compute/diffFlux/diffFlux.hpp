#ifndef __diffFlux_H__
#define __diffFlux_H__

#include "block_.hpp"

// ./diffFlux
//    |------> diffusiveFlux
void diffusiveFlux(block_ &b);
//    |------> alphaDamping
void alphaDampingFlux(block_ &b);

#endif

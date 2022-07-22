#ifndef __subgrid_H__
#define __subgrid_H__

#include "block_.hpp"

// ./subgrid
//    |------> mixedScaleModel
void mixedScaleModel(block_ b);
//    |------> smagorinsky
void smagorinsky(block_ b);

#endif

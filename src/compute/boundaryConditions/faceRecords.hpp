// One boundary face as a boundary condition sees it: its block's arrays, its
// own values, and where it sits.
#ifndef __faceRecords_H__
#define __faceRecords_H__

#include "kernelUtils.hpp"

struct faceRecords {
  unmanaged<double ****> q, Q, qh, S;
  unmanaged<double *****> grads;
  unmanaged<double ***> qBcVals, QBcVals;
  unmanaged<double **> rot;
  pgDims d;
  int nface;
};

#endif

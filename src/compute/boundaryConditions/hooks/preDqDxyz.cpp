#include "boundaryConditions/faceRecords.hpp"
#include "kernelUtils.hpp"

// The preDqDxyz hook of every boundary face, in one call. The jit includes the
// case's conditions that have this hook and lists them as an X-macro; each
// is a kind, and each face's kind picks its condition.
#ifndef PG_BCS
#error "a hook is compiled for one case: -DPG_BCS=\"X(name) ...\" from the jit"
#endif

#define X(name) name##Kind,
enum bcKind { PG_BCS };
#undef X

PG_ABI void pgPreDqDxyz(int count, pgOut *q_, pgOut *Q_, pgIn *qh_,
                        pgOut *grads_, pgIn *S_, pgIn *qBcVals_, pgIn *QBcVals_,
                        pgIn *rot_, const int *nface, const int *kind,
                        double tme) {
  for (int e = 0; e < count; e++) {
    const faceRecords face{as4(q_[e]),       as4(Q_[e]),     as4(qh_[e]),
                           as4(S_[e]),       as5(grads_[e]), as3(qBcVals_[e]),
                           as3(QBcVals_[e]), as2(rot_[e]),   nface[e]};
    switch (kind[e]) {
#define X(name)                                                                \
  case name##Kind:                                                             \
    name##_preDqDxyz(face, tme);                                               \
    break;
      PG_BCS
#undef X
    }
  }
}

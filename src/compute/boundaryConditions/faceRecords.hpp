// One boundary face as a boundary condition sees it: its block's arrays, its
// own values, which side of the block it is, and the planes of any array
// either side of it.
#ifndef __faceRecords_H__
#define __faceRecords_H__

#include "kernelUtils.hpp"

// A stack of planes of a block array, as a condition walks them: the layer
// first, then the two in-plane indices, then the components. Layer 0 is the
// plane next to the face and the layer stride carries the direction, so a
// halo runs outward and an interior inward with the same index.
template <class T, int R> struct planes {
  T *data;
  long stride[R];
  int extent[R];

  template <class... I> KOKKOS_INLINE_FUNCTION T &operator()(I... index) const {
    const long at[] = {static_cast<long>(index)...};
    long offset = 0;
    for (int d = 0; d < R; d++)
      offset += at[d] * stride[d];
    return data[offset];
  }
};

struct faceRecords {
  // what some hook writes is writable for all of them
  out4 q, Q;
  in4 qh, S;
  out5 grads;
  in3 qBcVals, QBcVals;
  in2 rot;
  int nface;

  // the axis this face is normal to, and whether it is the low end of it
  int axis() const { return (nface - 1) / 2; }
  bool low() const { return nface % 2 == 1; }
  // the sign of the outward normal along the axis
  double outward() const { return low() ? -1.0 : 1.0; }

  // `count` planes of `view` from the one at `start`, stepping `step` along
  // the axis; the in-plane axes keep their order
  template <class View>
  auto stack(const View &view, const int start, const int step,
             const int count) const {
    constexpr int rank = View::rank;
    planes<typename View::value_type, rank> p;
    p.data = view.data() + start * view.stride(axis());
    p.stride[0] = step * static_cast<long>(view.stride(axis()));
    p.extent[0] = count;
    int d = 1;
    for (int a = 0; a < rank; a++) {
      if (a == axis())
        continue;
      p.stride[d] = view.stride(a);
      p.extent[d] = view.extent(a);
      d++;
    }
    return p;
  }

  // the halo layers of a cell array, outward from the face
  template <class View> auto halo(const View &view) const {
    const int n = view.extent(axis());
    return stack(view, low() ? ng - 1 : n - ng, low() ? -1 : 1, ng);
  }
  // the interior layers of a cell array, inward from the face
  template <class View> auto interior(const View &view) const {
    const int n = view.extent(axis());
    return stack(view, low() ? ng : n - ng - 1, low() ? 1 : -1, ng);
  }
  // the one plane of a face array on the face itself: (i, j, components)
  template <class View> auto boundary(const View &view) const {
    constexpr int rank = View::rank;
    const int n = view.extent(axis());
    planes<typename View::value_type, rank - 1> p;
    p.data = view.data() + (low() ? ng : n - ng - 1) * view.stride(axis());
    int d = 0;
    for (int a = 0; a < rank; a++) {
      if (a == axis())
        continue;
      p.stride[d] = view.stride(a);
      p.extent[d] = view.extent(a);
      d++;
    }
    return p;
  }

  // every cell of `layers` planes: (layer, i, j), and with a component index
  MDRange3 range(const int layers) const {
    const int a = axis();
    const int i = a == 0 ? q.extent(1) : q.extent(0);
    const int j = a == 2 ? q.extent(1) : q.extent(2);
    return MDRange3({0, 0, 0}, {layers, i, j});
  }
  MDRange4 range(const int layers, const int components) const {
    const int a = axis();
    const int i = a == 0 ? q.extent(1) : q.extent(0);
    const int j = a == 2 ? q.extent(1) : q.extent(2);
    return MDRange4({0, 0, 0, 0}, {layers, i, j, components});
  }
};

#endif

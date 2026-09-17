// The launch: an item of a tile turned into a cell center, a cell face, a
// block face's halo cell or a halo exchange cell, a kernel struct pinned to
// it, and the shapes that run a body over every entry of a table. A kernel
// gets this through kernel.hpp.
#ifndef __launch_H__
#define __launch_H__

#include "arrays.hpp"

// A launch is tiles of items: a team does one tile of one entry's items.
// How many items a tile is comes with the tiling, the backend's knob for
// that kind of item: cells, or single elements. On a device a launch over
// cells or faces is bounded at PG_LAUNCH_THREADS with a floor of
// PG_LAUNCH_WAVES resident waves, the backend's knobs the jit bakes, which
// is the register budget the compiler sizes a body for: unbounded, it sizes
// for the largest team the device allows, and on an MI100 that floor of
// four spilled the viscous flux and transport (660 and 316 bytes a lane)
// while a floor of one starved the latency hiding; two halved both. A
// launch over elements has no registers to speak of and takes any tile.
#ifdef PG_LAUNCH_THREADS
using teams = Kokkos::TeamPolicy<
    execSpace, Kokkos::LaunchBounds<PG_LAUNCH_THREADS, PG_LAUNCH_WAVES>>;
#else
using teams = Kokkos::TeamPolicy<execSpace>;
#endif
using elementTeams = Kokkos::TeamPolicy<execSpace>;
using team = teams::member_type;

// a team's tile: its entry and its item range
struct tile {
  int e, begin, end;
};
KOKKOS_INLINE_FUNCTION tile tileOf(const pgTiling &t, const team &m) {
  const int rank = m.league_rank();
  const int e = t.entry[rank];
  const int begin = (rank - t.first[e]) * t.tile;
  const int n = t.items[e];
  return {e, begin, begin + t.tile < n ? begin + t.tile : n};
}
// a cell team is its tile, one thread per item, as far as the space allows:
// the whole tile on a device, one thread on the host
template <class F> inline teams policyOf(const pgTiling &t, const F &body) {
  const int most = teams(1, 1).team_size_max(body, Kokkos::ParallelForTag());
  return teams(t.tiles, t.tile < most ? t.tile : most);
}
template <class F, class R>
inline teams policyOf(const pgTiling &t, const F &body, const R &reducer) {
  const int most =
      teams(1, 1).team_size_max(body, reducer, Kokkos::ParallelReduceTag());
  return teams(t.tiles, t.tile < most ? t.tile : most);
}
inline elementTeams elementPolicyOf(const pgTiling &t) {
  return elementTeams(t.tiles, Kokkos::AUTO);
}

// an item as indices; the layout's fastest index runs fastest
template <int R>
KOKKOS_INLINE_FUNCTION void unravel(int item, const int *extent, int *index) {
  if constexpr (std::is_same_v<layout, Kokkos::LayoutLeft>) {
    for (int d = 0; d < R; d++) {
      index[d] = item % extent[d];
      item /= extent[d];
    }
  } else {
    for (int d = R - 1; d >= 0; d--) {
      index[d] = item % extent[d];
      item /= extent[d];
    }
  }
}

// an item as indices in a fixed axis order, the first running fastest; the
// order is compiled in, so every index stays in a register
template <int R, int... O>
KOKKOS_INLINE_FUNCTION void unravelAs(int item, const int *extent, int *index) {
  constexpr int order[] = {O...};
  for (int d = 0; d < R; d++) {
    index[order[d]] = item % extent[order[d]];
    item /= extent[order[d]];
  }
}

// an item of a face's planes as its cell (layer, a, b), walking the block's
// memory contiguously: on the left layout that is the block's first axis,
// which is the layer on an i face and `a` on a j or k face; on the right
// layout the natural order backwards. The components are looped in the
// thread, so the decode is paid once per cell, not once per value.
KOKKOS_INLINE_FUNCTION void unravelPlanes(int item, const int *extent,
                                          const int axis, int *index) {
  if constexpr (std::is_same_v<layout, Kokkos::LayoutLeft>) {
    if (axis == 0)
      unravelAs<3, 0, 1, 2>(item, extent, index);
    else if (axis == 1)
      unravelAs<3, 1, 0, 2>(item, extent, index);
    else
      unravelAs<3, 1, 2, 0>(item, extent, index);
  } else {
    unravelAs<3, 2, 1, 0>(item, extent, index);
  }
}

// an item of one entry's cells turned back into indices. On the left
// layout a face kernel walks its own direction right after the fastest
// index, so the plane a face reads on its far side is the one the tile
// before walked and still in cache: measured on an MI100, the k viscous
// flux 5.6 -> 4.3 ms at 12 species, the j direction already walks so
#if defined(PG_DIRECTION) && PG_DIRECTION == 2
constexpr int walkNext = 2;
#else
constexpr int walkNext = 1;
#endif
KOKKOS_INLINE_FUNCTION void cellAt(const pgCells &c, const int item, int &i,
                                   int &j, int &k) {
  int a[3];
  if constexpr (std::is_same_v<layout, Kokkos::LayoutLeft>)
    unravelAs<3, 0, walkNext, 3 - walkNext>(item, c.extent, a);
  else
    unravel<3>(item, c.extent, a);
  i = c.start[0] + a[0], j = c.start[1] + a[1], k = c.start[2] + a[2];
}
KOKKOS_INLINE_FUNCTION void cellAt(const pgCells &c, const int item, int &i,
                                   int &j, int &k, int &l) {
  int a[4];
  unravel<4>(item, c.extent, a);
  i = c.start[0] + a[0], j = c.start[1] + a[1], k = c.start[2] + a[2], l = a[3];
}

// A kernel is an aggregate of its members. The launch shape pins every
// column and dims member to the cell before calling the body, walking the
// members by count with a structured binding; there is no other way to
// reach a struct's members generically before C++26.
struct anything {
  template <class T> operator T() const;
};
template <class K, class... A> constexpr int arityFrom() {
  if constexpr (requires { K{A{}..., anything{}}; })
    return arityFrom<K, A..., anything>();
  else
    return sizeof...(A);
}
template <class K> constexpr int arity = arityFrom<K>();

// what pins, and what does not; a member pins to the position its kind
// takes (a block column to a cell, a face column to a plane)
template <class M, class P> KOKKOS_INLINE_FUNCTION void pin(M &, const P &) {}
template <class M, class P>
  requires requires(M &m, const P &p) { m.pin(p); }
KOKKOS_INLINE_FUNCTION void pin(M &m, const P &p) {
  m.pin(p);
}
template <class P, class... M>
KOKKOS_INLINE_FUNCTION void pinEach(const P &at, M &...m) {
  (pin(m, at), ...);
}
template <class K, class P>
KOKKOS_INLINE_FUNCTION void pinAll(K &kernel, const P &at) {
  constexpr int n = arity<K>;
  static_assert(n <= 16, "a kernel has at most 16 members");
  if constexpr (n == 1) {
    auto &[m1] = kernel;
    pinEach(at, m1);
  } else if constexpr (n == 2) {
    auto &[m1, m2] = kernel;
    pinEach(at, m1, m2);
  } else if constexpr (n == 3) {
    auto &[m1, m2, m3] = kernel;
    pinEach(at, m1, m2, m3);
  } else if constexpr (n == 4) {
    auto &[m1, m2, m3, m4] = kernel;
    pinEach(at, m1, m2, m3, m4);
  } else if constexpr (n == 5) {
    auto &[m1, m2, m3, m4, m5] = kernel;
    pinEach(at, m1, m2, m3, m4, m5);
  } else if constexpr (n == 6) {
    auto &[m1, m2, m3, m4, m5, m6] = kernel;
    pinEach(at, m1, m2, m3, m4, m5, m6);
  } else if constexpr (n == 7) {
    auto &[m1, m2, m3, m4, m5, m6, m7] = kernel;
    pinEach(at, m1, m2, m3, m4, m5, m6, m7);
  } else if constexpr (n == 8) {
    auto &[m1, m2, m3, m4, m5, m6, m7, m8] = kernel;
    pinEach(at, m1, m2, m3, m4, m5, m6, m7, m8);
  } else if constexpr (n == 9) {
    auto &[m1, m2, m3, m4, m5, m6, m7, m8, m9] = kernel;
    pinEach(at, m1, m2, m3, m4, m5, m6, m7, m8, m9);
  } else if constexpr (n == 10) {
    auto &[m1, m2, m3, m4, m5, m6, m7, m8, m9, m10] = kernel;
    pinEach(at, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10);
  } else if constexpr (n == 11) {
    auto &[m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11] = kernel;
    pinEach(at, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11);
  } else if constexpr (n == 12) {
    auto &[m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12] = kernel;
    pinEach(at, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12);
  } else if constexpr (n == 13) {
    auto &[m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13] = kernel;
    pinEach(at, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13);
  } else if constexpr (n == 14) {
    auto &[m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14] =
        kernel;
    pinEach(at, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14);
  } else if constexpr (n == 15) {
    auto &[m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15] =
        kernel;
    pinEach(at, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14,
            m15);
  } else if constexpr (n == 16) {
    auto &[m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15,
           m16] = kernel;
    pinEach(at, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14,
            m15, m16);
  }
}
// a copy of the kernel pinned to a position
template <class K, class P>
KOKKOS_INLINE_FUNCTION K pinned(const K &k, const P &at) {
  K p = k;
  pinAll(p, at);
  return p;
}

// The launch shapes. A kernel is a struct: its arguments as members and its
// body on one item as operator(); the entry point hands it to the shape it
// is. Each shape is the same ten lines, written once, with no branch in it.
// The struct has to be a named type: the NVIDIA compiler will not put a
// kernel lambda in a template instantiated with another lambda's type.

// a launch, or the next node of the graph under capture
template <class P, class B>
void launch(const char *name, const P &policy, const B &body) {
  if (graphNode *tail = pgGraphTail())
    *tail = tail->then_parallel_for(std::string(name), policy, body);
  else
    Kokkos::parallel_for(name, policy, body);
}
// a reduction is read on the host as it returns: it has no place in a graph
inline void notCapturing(const char *name) {
  if (pgGraphTail())
    Kokkos::abort(
        (std::string(name) + ": a reduction under graph capture").c_str());
}

// a body on every cell of every entry, its columns pinned to the cell
template <class F>
void forCells(const char *name, const pgTiling &t, const F &f) {
  auto body = KOKKOS_LAMBDA(const team &team) {
    const auto r = tileOf(t, team);
    const auto p = pinned(f, entry{r.e});
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, r.begin, r.end),
                         [&](const int item) {
                           cell c{r.e};
                           cellAt(t.cells[r.e], item, c.i, c.j, c.k);
                           pinned(p, within{c})();
                         });
  };
  launch(name, policyOf(t, body), body);
}

// a body on every element of every entry's allocation, halos and all,
// f(i) with the columns pinned to the entry and read flat, A[i]: what a
// copy or a linear combination of whole arrays is, and nothing else. The
// range is PG_RANGE(elements, components = ...), the arrays' width.
template <class F>
void forElements(const char *name, const pgTiling &t, const F &f) {
  auto body = KOKKOS_LAMBDA(const team &team) {
    const auto r = tileOf(t, team);
    const auto p = pinned(f, cell{r.e, 0, 0, 0});
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, r.begin, r.end),
                         [&](const int item) { p(item); });
  };
  launch(name, elementPolicyOf(t), body);
}

// a body on every cell and component: f(l)
template <class F>
void forCellsAndComponents(const char *name, const pgTiling &t, const F &f) {
  auto body = KOKKOS_LAMBDA(const team &team) {
    const auto r = tileOf(t, team);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, r.begin, r.end),
                         [&](const int item) {
                           cell c{r.e};
                           int l;
                           cellAt(t.cells[r.e], item, c.i, c.j, c.k, l);
                           pinned(f, c)(l);
                         });
  };
  launch(name, elementPolicyOf(t), body);
}

// a reduction over every cell: f(value &), joined within each team and then
// across them, into what the reducer holds
template <class F, class R>
void reduceCells(const char *name, const pgTiling &t, const F &f,
                 const R &reducer) {
  notCapturing(name);
  using value = typename R::value_type;
  auto body = KOKKOS_LAMBDA(const team &team, value &upd) {
    const auto r = tileOf(t, team);
    value mine;
    reducer.init(mine);
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, r.begin, r.end),
        [&](const int item, value &v) {
          cell c{r.e};
          cellAt(t.cells[r.e], item, c.i, c.j, c.k);
          pinned(f, c)(v);
        },
        R(mine));
    Kokkos::single(Kokkos::PerTeam(team), [&]() { reducer.join(upd, mine); });
  };
  Kokkos::parallel_reduce(name, policyOf(t, body, reducer), body, reducer);
}

// the same over every cell and component: f(l, value &)
template <class F, class R>
void reduceCellsAndComponents(const char *name, const pgTiling &t, const F &f,
                              const R &reducer) {
  notCapturing(name);
  using value = typename R::value_type;
  Kokkos::parallel_reduce(
      name, elementPolicyOf(t),
      KOKKOS_LAMBDA(const team &team, value &upd) {
        const auto r = tileOf(t, team);
        value mine;
        reducer.init(mine);
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, r.begin, r.end),
            [&](const int item, value &v) {
              cell c{r.e};
              int l;
              cellAt(t.cells[r.e], item, c.i, c.j, c.k, l);
              pinned(f, c)(l, v);
            },
            R(mine));
        Kokkos::single(Kokkos::PerTeam(team),
                       [&]() { reducer.join(upd, mine); });
      },
      reducer);
}

// one condition's body on every halo cell of every face in the table
template <class F>
void forBlockFacePlanes(const char *name, const pgTiling &t, const F &f,
                        const int *nface) {
  auto body = KOKKOS_LAMBDA(const team &team) {
    const auto r = tileOf(t, team);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, r.begin, r.end),
                         [&](const int item) {
                           plane p{r.e, 0, 0, 0, nface[r.e]};
                           cellAt(t.cells[r.e], item, p.g, p.i, p.j);
                           pinned(f, p)();
                         });
  };
  launch(name, policyOf(t, body), body);
}

// a halo exchange's body on every plane cell of every layer of every block
// face in the table, walked in the block's memory order
template <class F>
void forHaloExchange(const char *name, const pgTiling &t, const F &f,
                     const int *nface) {
  auto body = KOKKOS_LAMBDA(const team &team) {
    const auto r = tileOf(t, team);
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, r.begin, r.end), [&](const int item) {
          int at[3];
          unravelPlanes(item, t.cells[r.e].extent, faceAxis(nface[r.e]), at);
          pinned(f, plane{r.e, at[0], at[1], at[2], nface[r.e]})();
        });
  };
  launch(name, elementPolicyOf(t), body);
}

#endif

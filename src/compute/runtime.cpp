#include "abi.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <vector>

// The device graphs a run captures: a flow's launches, recorded once as a
// chain of nodes in the order they were made and submitted as one from
// then on. One capture is open at a time; the launch shapes read its tail.
namespace {
struct capturedGraph {
  Kokkos::Experimental::Graph<execSpace> graph;
  graphNode tail;
  capturedGraph() : tail(graph.root_node()) {}
};
// leaked on purpose: a Kokkos object destroyed after finalize aborts, and
// pgFinalize clears this first
auto &graphs = *new std::vector<std::unique_ptr<capturedGraph>>;
capturedGraph *capturing = nullptr;
} // namespace

// The runtime half of the ABI: memory where the kernels run, and the copies
// in and out of it.
PG_ABI void pgInitialize() {
  if (!Kokkos::is_initialized())
    Kokkos::initialize();
}

PG_ABI void pgFinalize() {
  graphs.clear();
  capturing = nullptr;
  if (Kokkos::is_initialized())
    Kokkos::finalize();
}

// 1 if arrays are LayoutLeft on this device, so the host side allocates to
// match
PG_ABI int pgLayoutLeft() {
  return std::is_same<layout, Kokkos::LayoutLeft>::value;
}

// the execution space the runtime was built for, as Kokkos names it:
// Serial, OpenMP, Cuda, HIP
PG_ABI const char *pgBackend() { return execSpace::name(); }

// 1 if the kernels' memory is host memory, so python can hand a kernel its
// own numpy buffers. Not SpaceAccessibility: a managed-memory build is
// accessible from the host, but numpy's allocations are not managed.
PG_ABI int pgOnHost() { return std::is_same_v<viewSpace, Kokkos::HostSpace>; }

// zeroed, as a Kokkos::View would be: the kernels accumulate into fresh arrays
PG_ABI void *pgAllocate(size_t bytes) {
  void *device = Kokkos::kokkos_malloc<viewSpace>("pg", bytes);
  using deviceBytes =
      Kokkos::View<char *, viewSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  Kokkos::deep_copy(deviceBytes(static_cast<char *>(device), bytes), 0);
  return device;
}

// opens a capture: every launch until pgGraphEnd becomes a node; the id
// submits it later
PG_ABI int pgGraphBegin() {
  graphs.push_back(std::make_unique<capturedGraph>());
  capturing = graphs.back().get();
  return static_cast<int>(graphs.size()) - 1;
}

PG_ABI void pgGraphEnd() {
  capturing->graph.instantiate();
  capturing = nullptr;
}

// the graph runs in order on the execution space, after what was queued
// before it, like the launches it stands for
PG_ABI void pgGraphSubmit(int id) { graphs[id]->graph.submit(execSpace{}); }

// a graph whose arrays or faces changed is dropped; a new capture replaces it
PG_ABI void pgGraphDrop(int id) { graphs[id].reset(); }

PG_ABI graphNode *pgGraphTail() {
  return capturing ? &capturing->tail : nullptr;
}

// an array outliving finalize is the process exiting; there is nothing to free
PG_ABI void pgFree(void *device) {
  if (Kokkos::is_initialized())
    Kokkos::kokkos_free<viewSpace>(device);
}

// A copy is queued on the execution space like a kernel, so a device-side
// consumer follows it in order; only a host-side reader has to wait.
PG_ABI void pgToHost(const void *device, void *host, size_t bytes, int wait) {
  using deviceBytes = Kokkos::View<const char *, viewSpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using hostBytes =
      Kokkos::View<char *, hostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  Kokkos::deep_copy(execSpace(), hostBytes(static_cast<char *>(host), bytes),
                    deviceBytes(static_cast<const char *>(device), bytes));
  if (wait)
    Kokkos::fence();
}

PG_ABI void pgToDevice(const void *host, void *device, size_t bytes, int wait) {
  using hostBytes = Kokkos::View<const char *, hostSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using deviceBytes =
      Kokkos::View<char *, viewSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  Kokkos::deep_copy(execSpace(),
                    deviceBytes(static_cast<char *>(device), bytes),
                    hostBytes(static_cast<const char *>(host), bytes));
  if (wait)
    Kokkos::fence();
}

PG_ABI void pgFence() { Kokkos::fence(); }

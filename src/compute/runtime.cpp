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
// an open fork: the node its siblings hang off, the end of each sibling
// closed so far, and whether one is open
struct openFork {
  graphNode base;
  std::vector<graphNode> ends;
  bool open = false;
};
auto &forks = *new std::vector<openFork>;
// leaked on purpose: a Kokkos object destroyed after finalize aborts, and
// pgFinalize clears this first
auto &graphs = *new std::vector<std::unique_ptr<capturedGraph>>;
capturedGraph *capturing = nullptr;
// host memory the device reaches at bus speed, what an exchange stages its
// messages through: kept for the run, freed at finalize
auto &pinned = *new std::vector<void *>;
} // namespace

// A second execution space instance for the copies an exchange stages
// through the host, so they run beside the kernels instead of between
// them: a copy out is ordered behind whatever the kernels' instance has
// queued (the pack), the host waits on the copies alone, and a copy in
// is waited on before the kernel that reads it is launched. On a host
// build the instances are one and the copies synchronous.
namespace {
// made on first use and destroyed by pgFinalize: an instance outliving
// finalize aborts at exit
execSpace *copyInstance = nullptr;
execSpace &copySpace() {
  if (!copyInstance) {
#if defined(KOKKOS_ENABLE_HIP)
    hipStream_t stream;
    KOKKOS_IMPL_HIP_SAFE_CALL(
        hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    copyInstance = new execSpace(stream);
#elif defined(KOKKOS_ENABLE_CUDA)
    cudaStream_t stream;
    KOKKOS_IMPL_CUDA_SAFE_CALL(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    copyInstance = new execSpace(stream);
#else
    copyInstance = new execSpace();
#endif
  }
  return *copyInstance;
}
// the copy instance waits for what the kernels' instance has queued so far
void copyAfterKernels() {
#if defined(KOKKOS_ENABLE_HIP)
  static hipEvent_t event = [] {
    hipEvent_t e;
    KOKKOS_IMPL_HIP_SAFE_CALL(
        hipEventCreateWithFlags(&e, hipEventDisableTiming));
    return e;
  }();
  KOKKOS_IMPL_HIP_SAFE_CALL(hipEventRecord(event, execSpace().hip_stream()));
  KOKKOS_IMPL_HIP_SAFE_CALL(
      hipStreamWaitEvent(copySpace().hip_stream(), event, 0));
#elif defined(KOKKOS_ENABLE_CUDA)
  static cudaEvent_t event = [] {
    cudaEvent_t e;
    KOKKOS_IMPL_CUDA_SAFE_CALL(
        cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    return e;
  }();
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventRecord(event, execSpace().cuda_stream()));
  KOKKOS_IMPL_CUDA_SAFE_CALL(
      cudaStreamWaitEvent(copySpace().cuda_stream(), event, 0));
#endif
}
} // namespace

// The runtime half of the ABI: memory where the kernels run, and the copies
// in and out of it.
PG_ABI void pgInitialize() {
  if (!Kokkos::is_initialized())
    Kokkos::initialize();
}

PG_ABI void pgFinalize() {
  graphs.clear();
  forks.clear();
  capturing = nullptr;
  delete copyInstance;
  copyInstance = nullptr;
  for (void *host : pinned)
    Kokkos::kokkos_free<Kokkos::SharedHostPinnedSpace>(host);
  pinned.clear();
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

// Independent launches under capture: a fork remembers the tail, each
// sibling's launches hang off it, and the join is a node after all of them
// (when_all), so the device may run the siblings in any order or together.
// Without a capture they are launches in order like any others.
PG_ABI void pgGraphFork() {
  if (capturing)
    forks.push_back({capturing->tail, {}});
}

PG_ABI void pgGraphSibling() {
  if (!capturing)
    return;
  openFork &fork = forks.back();
  if (fork.open)
    fork.ends.push_back(capturing->tail);
  capturing->tail = fork.base;
  fork.open = true;
}

PG_ABI void pgGraphJoin() {
  if (!capturing)
    return;
  openFork &fork = forks.back();
  if (fork.open)
    fork.ends.push_back(capturing->tail);
  graphNode joined = fork.ends.empty() ? fork.base : fork.ends[0];
  for (size_t i = 1; i < fork.ends.size(); ++i)
    joined = Kokkos::Experimental::when_all(joined, fork.ends[i]);
  capturing->tail = joined;
  forks.pop_back();
}

// an array outliving finalize is the process exiting; there is nothing to free
PG_ABI void pgFree(void *device) {
  if (Kokkos::is_initialized())
    Kokkos::kokkos_free<viewSpace>(device);
}

PG_ABI void pgCopyToHostAside(const void *device, void *host, size_t bytes) {
  using deviceBytes = Kokkos::View<const char *, viewSpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using hostBytes =
      Kokkos::View<char *, hostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  copyAfterKernels();
  Kokkos::deep_copy(copySpace(), hostBytes(static_cast<char *>(host), bytes),
                    deviceBytes(static_cast<const char *>(device), bytes));
}
PG_ABI void pgCopyToDeviceAside(const void *host, void *device, size_t bytes) {
  using hostBytes = Kokkos::View<const char *, hostSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using deviceBytes =
      Kokkos::View<char *, viewSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  Kokkos::deep_copy(copySpace(),
                    deviceBytes(static_cast<char *>(device), bytes),
                    hostBytes(static_cast<const char *>(host), bytes));
}
// the host waits for the copies alone
PG_ABI void pgCopyWait() { copySpace().fence(); }

// pinned host memory, on a host build host memory like any other
PG_ABI void *pgAllocatePinned(size_t bytes) {
  pinned.push_back(
      Kokkos::kokkos_malloc<Kokkos::SharedHostPinnedSpace>("pgPinned", bytes));
  return pinned.back();
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

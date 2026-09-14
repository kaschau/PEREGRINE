#include "abi.hpp"
#include <Kokkos_Core.hpp>

// The runtime half of the ABI: memory where the kernels run, and the copies
// in and out of it.
PG_ABI void pgInitialize() {
  if (!Kokkos::is_initialized())
    Kokkos::initialize();
}

PG_ABI void pgFinalize() {
  if (Kokkos::is_initialized())
    Kokkos::finalize();
}

// 1 if arrays are LayoutLeft on this device, so the host side allocates to
// match
PG_ABI int pgLayoutLeft() {
  return std::is_same<layout, Kokkos::LayoutLeft>::value;
}

// zeroed, as a Kokkos::View would be: the kernels accumulate into fresh arrays
PG_ABI void *pgAllocate(size_t bytes) {
  void *device = Kokkos::kokkos_malloc<viewSpace>("pg", bytes);
  using deviceBytes =
      Kokkos::View<char *, viewSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  Kokkos::deep_copy(deviceBytes(static_cast<char *>(device), bytes), 0);
  return device;
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

PG_ABI void pgCopy(void *dst, const void *src, size_t bytes) {
  using dstBytes =
      Kokkos::View<char *, viewSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using srcBytes = Kokkos::View<const char *, viewSpace,
                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  Kokkos::deep_copy(execSpace(), dstBytes(static_cast<char *>(dst), bytes),
                    srcBytes(static_cast<const char *>(src), bytes));
}

PG_ABI void pgFence() { Kokkos::fence(); }

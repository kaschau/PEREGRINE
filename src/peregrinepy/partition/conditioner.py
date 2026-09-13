from .mergeMixin import MergeMixin


class Conditioner(MergeMixin):
    """Readies a freshly translated grid for everything downstream. A
    translator wants this and nothing else, so it is its own small class
    rather than a partitioner nobody is going to partition with."""

    def condition(self, mb):
        """Ready a freshly translated grid: work out which of its interfaces
        are really periodic, merge away every interface the grid does not
        need, relabel every block so its longest extent is i, and make the
        interfaces that are left agree to the last digit."""
        print("Conditioning the grid...")
        before = len(mb.blocks)
        found = mb.detectPeriodics()
        removed = self.mergeAll(mb)
        self.longestAxisFirst(mb)
        off = self.matchInterfaces(mb)
        for kind, n in sorted(found.items()):
            print(f"  found {n} {kind} face(s) among the interfaces")
        print(
            f"  merged away {removed} interface(s), {before} blocks -> {len(mb.blocks)}"
        )
        print("  every block re-indexed so its longest extent is i")
        print(f"  interfaces matched exactly, closing a gap of up to {off:.3e}")

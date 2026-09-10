from .blockOpsMixin import BlockOpsMixin


class Conditioner(BlockOpsMixin):
    """Readies a freshly translated grid for everything downstream. A
    translator wants this and nothing else, so it is its own small class
    rather than a partitioner nobody is going to partition with."""

    def condition(self, mb):
        """Ready a freshly translated grid: merge away every interface the grid
        does not need, then relabel every block so its longest extent is i."""
        print("Conditioning the grid...")
        before = len(mb)
        removed = self.mergeAll(mb)
        self.longestAxisFirst(mb)
        print(f"  merged away {removed} interface(s), {before} blocks -> {len(mb)}")
        print("  every block re-indexed so its longest extent is i")

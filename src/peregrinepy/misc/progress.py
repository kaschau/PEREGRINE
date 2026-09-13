"""A progress bar over a loop."""

import sys


class Progress:
    """A bar toward :total:, redrawn on every report and ended when it gets
    there -- or, in a with block, when the block does. A quiet one draws
    nothing, so a caller with nowhere to show a bar need not ask."""

    length = 31
    dude = "¯\\_(ツ)_/¯"

    def __init__(self, total, quiet=False):
        self.total, self.quiet, self.done = total, quiet, 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if not self.quiet and self.done < self.total:
            sys.stdout.write("\n")

    def step(self, note=""):
        """One more of the total done."""
        self.at(self.done + 1, note)

    def at(self, done, note=""):
        """The bar at :done: of the total."""
        self.done = done
        if self.quiet:
            return
        fraction = min(done / float(self.total), 1.0)
        completed = int(round(self.length * fraction))
        bar = "_" * completed + self.dude + "_" * (self.length - completed)
        end = "\n" if fraction >= 1.0 else "\r"
        sys.stdout.write(f"[{bar}] {round(100.0 * fraction, 1)}% ...{note}{end}")
        sys.stdout.flush()

"""
Where a block's cells are. Every block that holds coordinates can work this
out; a solver extends it with the face vectors a flux is taken through.
"""


class MetricsMixin:
    def computeMetrics(self):
        """Where this block's cells are. A block with no solution on it has no
        use for the face vectors a flux is taken through."""
        if self.array["x"] is None:
            raise ValueError(
                "You must initialize the grid arrays before computing metrics"
            )

        # a cell center is the mean of the eight corners around it
        for c in "xyz":
            v = self.array[c]
            self.array[f"{c}c"][:] = 0.125 * (
                v[0:-1, 0:-1, 0:-1]
                + v[0:-1, 0:-1, 1::]
                + v[0:-1, 1::, 0:-1]
                + v[0:-1, 1::, 1::]
                + v[1::, 0:-1, 0:-1]
                + v[1::, 0:-1, 1::]
                + v[1::, 1::, 0:-1]
                + v[1::, 1::, 1::]
            )

        self.updateDeviceView([f"{c}c" for c in "xyz"])

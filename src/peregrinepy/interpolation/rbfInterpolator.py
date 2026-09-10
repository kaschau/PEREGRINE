from scipy import interpolate

from .baseInterpolator import BaseInterpolator


class RbfInterpolator(BaseInterpolator):
    """A radial basis function through every sample. :function: is which one
    -- linear, cubic, multiquadric and so on -- and :smooth: how closely it
    is held to the samples, zero being through them exactly."""

    interpolatorName = "rbf"

    def __init__(self, function="linear", smooth=0.5, **kwargs):
        super().__init__(**kwargs)
        self.function = function
        self.smooth = smooth

    def prepare(self, fromPts, toPts):
        """Scipy solves for the basis weights against the values, so unlike a
        nearest neighbour's tree this cannot be built from the points alone."""

        def onto(values):
            rbf = interpolate.Rbf(
                *fromPts.T, values, function=self.function, smooth=self.smooth
            )
            return rbf(*toPts.T)

        return onto

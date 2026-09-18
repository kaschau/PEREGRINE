from .baseInterpolator import BaseInterpolator


class NearestInterpolator(BaseInterpolator):
    """Every point takes the value of the sample nearest it. Cheap, and it
    invents nothing, but it is not smooth across the seam."""

    name = "nearest"
    # every value is one it was given, so there is no new extremum to clip
    canOvershoot = False

    def prepare(self, fromPts, toPts):
        # the tree is built once and thrown away, so the cheap build is worth
        # more than the tighter one, and the query threads
        from scipy import spatial

        tree = spatial.cKDTree(fromPts, balanced_tree=False, compact_nodes=False)
        _, nearest = tree.query(toPts, workers=-1)
        return lambda values: values[nearest]

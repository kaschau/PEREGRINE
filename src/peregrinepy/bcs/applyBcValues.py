"""
Putting a case onto the faces the grid left for it.

The grid names a face and says nothing more about it, because the same grid
runs as a wall on one case and an inlet on the next. The config says what each
name is and what it reads, and this puts both on the face.
"""

from .base import getBc, prep


def applyBcValues(mb):
    """Make every named face what its config entry says, and give it the
    values that entry sets."""
    bcValues = mb.config["bcValues"]

    for blk in mb:
        for face in blk.faces:
            if face.bcName is None:
                continue

            if face.bcName not in bcValues:
                raise KeyError(
                    f"block {blk.nblki} face {face.nface} carries the name"
                    f" '{face.bcName}', which this config says nothing about."
                    f" It knows {sorted(bcValues)}."
                )
            entry = bcValues[face.bcName]
            if "bcType" not in entry:
                raise KeyError(f"bcValues entry '{face.bcName}' names no bcType.")

            face.bcType = entry["bcType"]
            if not getBc(face.bcType).values:
                continue

            # some bcs have prep work of their own, a constant mass flux or a
            # profile read off disk, so they are asked rather than assigned to
            prep(blk, face, entry)

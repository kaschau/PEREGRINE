# PEREGRINE multiBlock #

## Blendin the Classes with C++ ##

This is really the heart of PEREGRINE. All the functionality of the code is
enabled by the structure created here. It is also where the C++/Kokkos library
merged with peregrinepy. 

The basic inheritance structure for a block/face object is:


           Topology
               |
               V
             Grid
               |
               V
            Restart
               |
               V
             Solver ---> self.cpp   <- block_ or face_. This comes from C++
                                       enabled by pybind11, and is held by
                                       the Solver rather than inherited.


In general, this works as expected. There are a few conventions to be aware of.

Consider the ```face``` object, and the variable ```nface``` which tells us
the face number. We want this defined at the topology level, and everything
that inherits the ```topologyFace```, as well as on the C++ side.

A name is a plain attribute where it is first defined, and a property
forwarding to ```self.cpp``` where there is a compute object to hold it.

**topologyFace**

``` python
    self.nface = nface  # this is just an int
```

**solverFace**

``` python
    @property
    def nface(self):
        return self.cpp.nface

    @nface.setter
    def nface(self, value):
        self.cpp.nface = value
```

so setting ```nface``` on a solver face lands it where a kernel will read it,
and the same name means the same thing at every level. A block does this with
```nblki```, ```ni```, ```nj```, ```nk```, ```ng``` and ```ne```. On the C++
side the member is spelled the same way.

``` c++
    int nface
```

Note the names carry no underscore. The Solver classes used to *inherit*
```block_```/```face_```, which put the python attribute and the C++ member in
the same slot: a property named ```nface``` would shadow the pybind11 member
of the same name, so the C++ member was named ```_nface``` to get out of the
way. The compute object is now held rather than inherited, so nothing can
shadow it and that workaround is gone. If you see an underscored name on the
compute side, it is a leftover and should be renamed.

Underscores on the python side are the ordinary private storage behind a
property that does real work -- ```_bcType``` behind the ```bcType``` check,
```_periodicRotation``` behind the setter that hands the matrix to the compute
side. Those have nothing to do with C++.

## Building of inherited setter methods ##

Sometime in the inheritance structure, we want to do more when we set the
class attribute than the inherited setter does. For example, when we set the
```periodicRotation``` of a ```topologyFace```, we just want to keep it. But a
```gridFace``` also wants it in ```array``` so the compute side has one of its
own, and a ```solverFace``` wants that pushed to the device on top. To do this
we define the setter at each level and build on the one below it.

**topologyFace**

``` python
    @periodicAxis.setter
    def periodicAxis(self, axis):
        self._periodicAxis = None if axis is None else axis / np.linalg.norm(axis)
```

**gridFace**
``` python
    @topologyFace.periodicRotation.setter
    def periodicRotation(self, rotation):
        # This calls the topology setter method
        topologyFace.periodicRotation.fset(self, rotation)

        # ... put it where the compute side reads it
```

**solverFace**
``` python
    @gridFace.periodicRotation.setter
    def periodicRotation(self, rotation):
        # This calls the grid setter method
        gridFace.periodicRotation.fset(self, rotation)

        # ... push it up to the device
```

Thus, setting the value of ```periodicRotation``` achieves the desired behavior at
each level of the inheritance structure.

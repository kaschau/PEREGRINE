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
               |    ___ block_ or face_  <-This comes from C++ 
               V   /                       enabled by pybind11
             Solver
             

In general, this works as expected. There are a few conventions to be aware of.

Consider the ```face``` object, and the variable ```nface``` which tells us the
face number. We want this defined at the topology level, and everything that
inherits the ```topologyFace```, as well as on the C++ side. But there are
restrictions on the value we can assign a face number (must be 1 through 6). So
we want to save users the hassle of accidentally setting an invalid face
number. But this creates a problem, since we also want the variable ```nface```
defined on the C++ side of things. And in C++ we must be specific about the type
of the variable ```nface``` (it is an int). So we cannot just go willy nillly on
the python side when creating the variable nface. We must ensure that the
variable on the python side that matches the variable on the C++ side is only
set to an integer. We cannot get fancy with python's dynamic typing because C++
wont have any of it. The solution to this is to use the following convention:

When a variable is meant to be set directly on the python side, and be reflected
on the C++ side, and the python <-> C++ link is not directly enabled with
pybind11 (for example int to int, float to float, etc.), and we want to use
setter/getter methods on the python side, we use underscores to denote the
variable names.

In the case of ```nface```, we decorate the name ```nface``` on the python side,
and store the end results in a variable ```_nface```.


So on the python side we create a setter method for the ```topologyFace```
object.

``` python
    self._nface  # This is just an int
    
    @property
    def nface(self):
        return self._nface

    @nface.setter(value):
        assert 1 <= value <= 6, "nface must be between 1 and 6"
        self._nface = value
```
        
Now, when we set a value of ```nface``` on the python side, it will go through
the setter method, perform the check on the value, and then apply the value to
the ```_nface``` attribute. When we access the ```nface``` attribute, the property
decorator just returns```_nface```. This will subsequently be updated on the C++ object
and accessible on the C++ side. Speaking of the C++ side, we use the underscored name.

``` c++
    int _nface
```

And then in C++ code we access the ```_nface``` variable no problem.

## Building of inherited setter methods ##

Sometime in the inheritance structure, we want to do more when we set the
class attribute than the inherited setter does. For example, when we set the
```periodixAxis``` values for a ```topologyFace```, we want to set the axis
values, normalize them if needed, and move on. But for a ```gridFace``` object,
we want to set the values *AND* calculate the ```periodicRotMatrixUp/Down```
arrays as well (if the periodic is rotational). But when we are a solver face,
we want to do all that *AND* create the corresponding Kokkos views/mirrors so
the rotational matrices are available on the C++ side. To do this we define the
setter that computes the matrix in the ```gridFace``` class, the build on it in
the solver setter for  ```periodicAxis``` as follows.

**topologyFace**

``` python
    @periodicAxis.setter
    def periodicAxis(self, axis):
        a = axis / np.linalg.norm(axis)
        self._periodicAxis = a
```

**gridFace**
``` python
    @topologyFace.periodicAxis.setter
    def periodicAxis(self, axis):
        # This calls the topology setter method
        topologyFace.periodicAxis.fset(self, axis) 
        
        # ... compute rotational matrix
```

**solverFace**
``` python
    @gridFace.periodicAxis.setter
    def periodicAxis(self, axis):
        # This calls the grid setter method
        gridFace.periodicAxis.fset(self, axis) 
        
        # ... set kokkos view/mirrors
```

Thus, setting the value of ```periodicAxis``` achieves the desired behavior at
each level of the inheritance structure.

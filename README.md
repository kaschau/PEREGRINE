# PEREGRINE

An attempt at a superfast Python/C++ Multi-Physics CFD code using Kokkos for performance portability. 


# Installation


``` setup.py install ```

Or just set PYTHONPATH to point to /path/to/PEREGRINE/src/peregrinepy
followed by manual install

``` mkdir build; cd build; ccmake ../```

To generate compile_commands.json, 

``` cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ../ ```


# Arrays


There are two means of manipulating data, on the python side as numpy arrays, and on the C++
side via Kokkos arrays. Python side arrays are accessed by a dictionary attribute
of the block class called "array", i.e.

    blk.array['q']

Gives access to the primative variables.

| Name       | Index | Variables       | Units        |
| :--------: | :---: | ---------       | -----        |
|**q**       |       | **Primatives**  |              |
|            |   0   | pressure        | Pa           |
|            | 1,2,3 | u,v,w           | m/s          |
|            |   4   | temperature     | K            |
|            | 5..ne | mass fraction   | []           |
|**Q**       |       | **Conserved**   |              |
|            |   0   | density         | kg/m^3       |
|            | 1,2,3 | momentum        | kg m / s.m^3 |
|            |   4   | total energy    | J            |
|            | 5..ne | species mass    | kg/m^3       |
|**qh**      |       | **Thermo**      |              |
|            |   0   | gamma           | []           |
|            |   1   | cp              | J/kg.K       |
|            |   2   | enthalpy        | J/m^3        |
|            |   3   | c               | m/s          |


## License

PEREGRINE is released under the New BSD License (see the LICENSE file for details).
Documentation is made available under a Creative Commons Attribution 4.0
license (see <http://creativecommons.org/licenses/by/4.0/>).

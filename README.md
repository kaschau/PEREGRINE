# PEREGRINE

An attempt at a superfast Python/C++ Multi-Physics CFD code using Kokkos for performance portability. 


Installation
------------

``` setup.py install ```

Or just set PYTHONPATH to point to /path/to/PEREGRINE/src/peregrinepy
followed by manual install

``` mkdir build; cd build; ccmake ../```

To generate compile_commands.json, 

``` cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ../ ```


## License

PEREGRINE is released under the New BSD License (see the LICENSE file for details).
Documentation is made available under a Creative Commons Attribution 4.0
license (see <http://creativecommons.org/licenses/by/4.0/>).

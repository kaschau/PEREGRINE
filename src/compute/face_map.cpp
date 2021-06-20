#include "kokkos_types.hpp"
#include "block_.hpp"
#include <string>
#include <iostream>

MDRange3 get_range3(block_ b, std::string face){


    MDRange3 range;

    if ( face.compare("0") == 0 ){
        // interior
        range = MDRange3({1,1,1},{b.ni,b.nj,b.nk});
    } else if ( face.compare("1") == 0 ) {
        // face 1 halo
        range = MDRange3({0,1,1},{1,b.nj,b.nk});
        //MDRange3 range({0,1,1},{0,b.nj,b.nk});
    } else if ( face.compare("2") == 0 ) {
        // face 2 halo
        range = MDRange3({b.ni,1,1},{b.ni+1,b.nj,b.nk});
    } else if ( face.compare("3") == 0 ) {
        // face 3 halo
        range = MDRange3({1,0,1},{b.ni,1,b.nk});
    } else if ( face.compare("4") == 0 ) {
        // face 4 halo
        range = MDRange3({1,b.nj,1},{b.ni,b.nj+1,b.nk});
    } else if ( face.compare("5") == 0 ) {
        // face 5 halo
        range = MDRange3({1,1,0},{b.ni,b.nj,1});
    } else if ( face.compare("6") == 0 ) {
        // face 6 halo
        range = MDRange3({1,1,b.nk},{b.ni,b.nj,b.nk+1});
    }

    return range;
}

#include "kokkos_types.hpp"
#include "block_.hpp"
#include <string>

MDRange3 get_range3(block_ b, std::string face){


    MDRange3 range;

    if ( face.compare("0") == 0 ){
        // interior
        range = MDRange3({1,1,1},{b.ni,b.nj,b.nk});
    } else if ( face.compare("1") == 0 ) {
        // face 1 halo
        range = MDRange3({0,0,0},{1,b.nj+1,b.nk+1});
        //MDRange3 range({0,1,1},{0,b.nj,b.nk});
    } else if ( face.compare("2") == 0 ) {
        // face 2 halo
        range = MDRange3({b.ni,0,0},{b.ni+1,b.nj+1,b.nk+1});
    } else if ( face.compare("3") == 0 ) {
        // face 3 halo
        range = MDRange3({0,0,0},{b.ni+1,1,b.nk+1});
    } else if ( face.compare("4") == 0 ) {
        // face 4 halo
        range = MDRange3({0,b.nj,0},{b.ni+1,b.nj+1,b.nk+1});
    } else if ( face.compare("5") == 0 ) {
        // face 5 halo
        range = MDRange3({0,0,0},{b.ni+1,b.nj+1,1});
    } else if ( face.compare("6") == 0 ) {
        // face 6 halo
        range = MDRange3({0,0,b.nk},{b.ni+1,b.nj+1,b.nk+1});
    }

    return range;
}

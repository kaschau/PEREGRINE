#include "kokkos_types.hpp"
#include "block_.hpp"
#include <string>

MDRange3 get_range3(block_ b, int face){


    MDRange3 range;

    if ( face == 0 ){
        // interior
        range = MDRange3({1,1,1},{b.ni,b.nj,b.nk});
    } else if ( face == 1 ) {
        // face 1 halo
        range = MDRange3({0,0,0},{1,b.nj+1,b.nk+1});
        //MDRange3 range({0,1,1},{0,b.nj,b.nk});
    } else if ( face == 2 ) {
        // face 2 halo
        range = MDRange3({b.ni,0,0},{b.ni+1,b.nj+1,b.nk+1});
    } else if ( face == 3 ) {
        // face 3 halo
        range = MDRange3({0,0,0},{b.ni+1,1,b.nk+1});
    } else if ( face == 4 ) {
        // face 4 halo
        range = MDRange3({0,b.nj,0},{b.ni+1,b.nj+1,b.nk+1});
    } else if ( face == 5 ) {
        // face 5 halo
        range = MDRange3({0,0,0},{b.ni+1,b.nj+1,1});
    } else if ( face == 6 ) {
        // face 6 halo
        range = MDRange3({0,0,b.nk},{b.ni+1,b.nj+1,b.nk+1});
    }

    return range;
}

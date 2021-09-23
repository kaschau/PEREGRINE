#include "kokkos_types.hpp"
#include "block_.hpp"
#include <stdexcept>

MDRange3 get_range3(block_ b,
                       int face,
                       int i/*=0*/,
                       int j/*=0*/,
                       int k/*=0*/){


    MDRange3 range;

    switch (face) {
      case -1 :
        // total block
        range = MDRange3({0,0,0},{b.ni+1,b.nj+1,b.nk+1});
        break;
      case 0 :
        // interior
        range = MDRange3({1,1,1},{b.ni,b.nj,b.nk});
        break;
      case 1 :
        // face 1 halo
        range = MDRange3({0,0,0},{1,b.nj+1,b.nk+1});
        break;
      case 2 :
        // face 2 halo
        range = MDRange3({b.ni,0,0},{b.ni+1,b.nj+1,b.nk+1});
        break;
      case 3 :
        // face 3 halo
        range = MDRange3({0,0,0},{b.ni+1,1,b.nk+1});
        break;
      case 4 :
        // face 4 halo
        range = MDRange3({0,b.nj,0},{b.ni+1,b.nj+1,b.nk+1});
        break;
      case 5 :
        // face 5 halo
        range = MDRange3({0,0,0},{b.ni+1,b.nj+1,1});
        break;
      case 6 :
        // face 6 halo
        range = MDRange3({0,0,b.nk},{b.ni+1,b.nj+1,b.nk+1});
        break;
      case 10 :
        // specify i,j,k turn it into a function call (kinda)
        range = MDRange3({i,j,k},{i+1,j+1,k+1});
        break;
      default :
        throw std::invalid_argument( "Unknown argument to get_range3");
    }

    return range;
}

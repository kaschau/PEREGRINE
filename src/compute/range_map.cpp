#include "kokkos_types.hpp"
#include "block_.hpp"
#include <stdexcept>

MDRange3 get_range3(block_ b,
                const int nface,
                const int i/*=0*/,
                const int j/*=0*/,
                const int k/*=0*/){


    MDRange3 range;

    switch (nface) {
      case -1 :
        // total block
        range = MDRange3({0, 0, 0}, {b.ni+2*b.ng-1, b.nj+2*b.ng-1, b.nk+2*b.ng-1});
        break;
      case 0 :
        // interior
        range = MDRange3({b.ng, b.ng, b.ng}, {b.ni+b.ng-1, b.nj+b.ng-1, b.nk+b.ng-1});
        break;
      case 1 :
        // face 1 halo
        range = MDRange3({0, 0, 0}, {b.ng, b.nj+2*b.ng-1, b.nk+2*b.ng-1});
        break;
      case 2 :
        // face 2 halo
        range = MDRange3({b.ni+b.ng-1, 0, 0}, {b.ni+2*b.ng-1, b.nj+2*b.ng-1, b.nk+2*b.ng-1});
        break;
      case 3 :
        // face 3 halo
        range = MDRange3({0, 0, 0}, {b.ni+2*b.ng-1, b.ng, b.nk+2*b.ng-1});
        break;
      case 4 :
        // face 4 halo
        range = MDRange3({0, b.nj+b.ng-1, 0}, {b.ni+2*b.ng-1, b.nj+2*b.ng-1, b.nk+2*b.ng-1});
        break;
      case 5 :
        // face 5 halo
        range = MDRange3({0, 0, 0}, {b.ni+2*b.ng-1, b.nj+2*b.ng-1, b.ng});
        break;
      case 6 :
        // face 6 halo
        range = MDRange3({0, 0, b.nk+b.ng-1}, {b.ni+2*b.ng-1, b.nj+2*b.ng-1, b.nk+2*b.ng-1});
        break;
      case 10 :
        // specify i,j,k turn it into a function call (kinda)
        range = MDRange3({i, j, k}, {i+1, j+1, k+1});
        break;
      default :
        throw std::invalid_argument( "Unknown argument to get_range3");
    }

    return range;
}

MDRange2 get_range2(block_ b,
                       int nface){

    MDRange2 range;

    switch (nface) {
      case 1: case 2:
        // face 1,2 range
        range = MDRange2({0, 0}, {b.nj+2*b.ng-1, b.nk+2*b.ng-1});
        break;
      case 3: case 4:
        // face 3,4 range
        range = MDRange2({0, 0}, {b.ni+2*b.ng-1, b.nk+2*b.ng-1});
        break;
      case 5: case 6:
        // face 5,6 range
        range = MDRange2({0, 0}, {b.ni+2*b.ng-1, b.nj+2*b.ng-1});
        break;
      default :
        throw std::invalid_argument( "Unknown argument to get_range2");
    }

    return range;
}

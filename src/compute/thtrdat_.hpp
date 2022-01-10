#ifndef __thtrdat__H__
#define __thtrdat__H__

#include "kokkos_types.hpp"
#include <vector>
#include <string>

// The struct that is sent to the Peregrine compute units. Holds all the
// thermodynamic and transport properties.
// Also converted into python class for modifying in the
// python wrapper
struct thtrdat_ {

  int ns;
  static constexpr double Ru=8314.46261815324;

  oneDview MW;

  // Constant cp values (reference cp)
  oneDview cp0;
  // NASA7 polynomial coefficients
  twoDview NASA7;

  // Generated temperature dependent viscosity poly'l coeff
  twoDview muPoly;
  // Generated temperature dependent thermal conductivity poly'l coeff
  twoDview kappaPoly;
  // Generated temperature dependent binary diffusion poly'l coeff
  // NOTE: pressure dependence is applied when the mixture diffusion
  // coefficients are created in-situ
  twoDview DijPoly;

  // Constant properties
  oneDview mu0;
  oneDview kappa0;

  // Critical properties;
  oneDview Tcrit;
  oneDview pcrit;
  oneDview Vcrit;
  oneDview acentric;
};

#endif

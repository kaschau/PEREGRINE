#ifndef __thtrdat__H__
#define __thtrdat__H__

#include "kokkos_types.hpp"
#include <string>
#include <vector>

// The struct that is sent to the Peregrine compute units. Holds all the
// thermodynamic and transport properties.
// Also converted into python class for modifying in the
// python wrapper
struct thtrdat_ {

  int ns;
  const double Ru = 8314.46261815324; // J/kmol/K

  // Molecular weight
  oneDview MW;

  // NASA7 polynomial coefficients
  twoDview NASA7;

  // Kinetic theory
  // Generated temperature dependent viscosity poly'l coeff
  twoDview muPoly;
  // Generated temperature dependent thermal conductivity poly'l coeff
  twoDview kappaPoly;
  // Generated temperature dependent binary diffusion poly'l coeff
  // NOTE: pressure dependence is applied when the mixture diffusion
  // coefficients are created in-situ
  twoDview DijPoly;

  // Constant properties
  oneDview cp0;
  oneDview mu0;
  oneDview kappa0;

  // Critical properties;
  oneDview Tcrit;
  oneDview pcrit;
  oneDview Vcrit;
  oneDview acentric;

  // Chung dense gas
  twoDview chungA;
  twoDview chungB;
  oneDview redDipole;
};

#endif

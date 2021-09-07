#ifndef __thtrdat__H__
#define __thtrdat__H__

#include <vector>
#include <string>

// The struct that is sent to the Peregrine compute units. Holds all the
// thermodynamic and transport properties.
// Also converted into python class for modifying in the
// python wrapper
struct thtrdat_ {

  int ns;
  double Ru;

  std::vector<std::string> species_names;
  std::vector<double> MW;

  // Constant cp values (reference cp)
  std::vector<double> cp0;
  // NASA7 polynomial coefficients
  std::vector<std::vector<double>> NASA7;

  // Generated temperature dependent viscosity poly'l coeff
  std::vector<std::vector<double>> mu_poly;
  // Generated temperature dependent thermal conductivity poly'l coeff
  std::vector<std::vector<double>> kappa_poly;
  // Generated temperature dependent binary diffusion poly'l coeff
  // NOTE: pressure dependence is applied when the mixture diffusion
  // coefficients are created in-situ
  std::vector<std::vector<double>> Dij_poly;

};

#endif

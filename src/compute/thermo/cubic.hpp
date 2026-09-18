// The cubic equation of state's root, shared by the fpdtype gas kernels.
#ifndef __cubic_H__
#define __cubic_H__

#include <Kokkos_Core.hpp>
#include <math.h>

// References
//
// Generalizing the Thermodynamics State Relationships in KIVA-3V
//     Mario F. Trujillo
//     Peter O’Rourke
//     David Torres
//     Los Alamos National Labs, 2002
//     https://www.osti.gov/servlets/purl/809947
//
// Thermodyanamic Properties from Cubic Equations of State
//     Patrick Chung-Nin Mak
//     University of British Columbia, 1988
//     https://open.library.ubc.ca/media/download/pdf/831/1.0058883/2
//
// Thermal and Transport Properties for the Simulation of Direct-Fired sCO 2
// Combustor
//     Manikantachari, Martin, Bobren-Diaz, Vasu
//     Journal of Engineering for Gas Turbines and Power, 2017
//     DOI: 10.1Real Gas Models in Coupled Algorithms Numerical
//

// Real Gas Models in Coupled Algorithms Numerical Recipes and Thermophysical
// Relations
//     Hanimann, Mangani, Casartelli, Vogt, Darwish
//     Turbomachinery Propulsion and Power, 2020
//     doi:10.3390/ijtpp5030020

// ----------------------------------------------------------------------//
//           Solves the cubic EOS of the general form
//
//                    RT         a \alpha(T)
//               P = ____   _   _______________
//
//                   V-b          V^2+2bV-b^2
//
// ----------------------------------------------------------------------//

// With three fpdtype roots, the stable phase is the one of least Gibbs energy.
// Taking the largest is correct only on the gas-like branch. A root at or
// below Bstar is not a physical volume, so it is not a candidate.
KOKKOS_INLINE_FUNCTION fpdtype stableRoot(const fpdtype x1, const fpdtype x2,
                                          const fpdtype x3, const fpdtype Astar,
                                          const fpdtype Bstar,
                                          const fpdtype uRG,
                                          const fpdtype wRG) {
  const fpdtype sq = sqrt(uRG * uRG - 4.0 * wRG);
  const fpdtype roots[3] = {x1, x2, x3};
  // if no root clears Bstar, fall back to the largest
  fpdtype Z = fmax(x1, fmax(x2, x3));
  fpdtype gMin = 0.0;
  bool found = false;
  for (int n = 0; n < 3; n++) {
    const fpdtype z = roots[n];
    if (z <= Bstar) {
      continue;
    }
    // Gibbs energy departure, g/(Ru T)
    const fpdtype g = z - 1.0 - log(z - Bstar) +
                      Astar / (Bstar * sq) *
                          log((2.0 * z + Bstar * (uRG - sq)) /
                              (2.0 * z + Bstar * (uRG + sq)));
    if (!found || g < gMin) {
      gMin = g;
      Z = z;
      found = true;
    }
  }
  return Z;
}

#endif

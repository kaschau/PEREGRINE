#ifndef __thermdat__H__
#define __thermdat__H__

#include <vector>
#include <string>

// The struct that is sent to the Peregrine compute units. Holds all the data arrays
// for each block. Also converted into python class for modifying in the
// python wrapper
struct thermdat_ {

  int ns;
  double R;

  std::vector<std::string> species_names;
  std::vector<double> MW;
  std::vector<double> cp0;

};

#endif

import numpy as np


def completeSpecies(key, usersp, refsp):
    # A function to collect the data in order of species listed in the input spdata
    # returns a numpy array of the data.
    prop = []
    for sp in usersp.keys():
        try:
            prop.append(usersp[sp][key])
        except KeyError:
            try:
                prop.append(refsp[sp][key])
            except KeyError:
                raise KeyError(
                    f"You want to use species {sp}, but did not provide a {key}, and it is not in the PEREGRINE species database."
                )
        except TypeError:
            try:
                prop.append(refsp[sp][key])
            except TypeError:
                raise TypeError(
                    "The top level in your spieces data input yaml file must only be species names."
                )
            except KeyError:
                raise KeyError(
                    f"You want to use species {sp}, but did not provide a {key}, and it is not in the PEREGRINE species database."
                )
    if type(prop[0]) == str:
        return prop
    else:
        return np.array(prop, dtype=np.float64)

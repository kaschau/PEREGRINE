import cantera as ct
from ..compute import thtrdat_
from pathlib import Path
import sys

class thtrdat(thtrdat_):

    def __init__(self, config):
        thtrdat_.__init__(self)

        relpath = str(Path(__file__).parent)
        ct.add_directory(relpath)
        gas = ct.Solution(config['thermochem']['ctfile'])

        ns = gas.n_species
        self.ns = ns
        self.Ru = ct.gas_constant

        # Species names string
        self.species_names = list(gas.species_names)

        # Species MW
        self.MW = list(gas.molecular_weights)


        #################################################################################
        ####### Set thermodynamic properties
        #################################################################################
        #Set either constant cp or NASA7 polynomial coefficients
        if config['thermochem']['eos'] == 'cpg':
            #Set gas to STP
            gas.TP = 293.15,101325.0
            # Values for constant Cp
            # J/(kg.K)
            cp0 = list(gas.standard_cp_R*ct.gas_constant/gas.molecular_weights)
            if len(cp0) != gas.n_species:
                raise ValueError('PEREGRINE ERROR: CPG info for all species (check cp0)')
            self.cp0 = cp0
        elif config['thermochem']['eos'] == 'tpg':
            N7 = []
            for n in range(gas.n_species):
                N7.append(list(gas.species()[n].thermo.coeffs))
            self.N7 = N7
        else:
            raise KeyError(f'PEREGRINE ERROR: Unknown EOS {config["thermochem"]["eos"]}')

        #################################################################################
        ####### Set transport properties
        #################################################################################

        if config['RHS']['diffusion']:
            from .MM_Tables import delta,tstar22,omega22_table,tstar,astar_table
            import numpy as np
            from scipy import interpolate as intrp

            deg = 5
            #Maximum and minumum temperatures to generate poly'l
            Tmin = 200
            Tmax = 3500
            #Generate range of temperatures
            npts = 50
            Ts = np.linspace(Tmin,Tmax,npts)

            # Collision integral interpolations
            intrp_o22 = intrp.RectBivariateSpline(tstar22, delta, omega22_table, kx=5,ky=5)
            intrp_Astar = intrp.RectBivariateSpline(tstar, delta, astar_table, kx=5,ky=5)

            # Get properties
            MW = gas.molecular_weights
            mass = MW/ct.avogadro
            kb = ct.boltzmann
            Ru = ct.gas_constant
            eps0 = ct.epsilon_0

            #epsilon
            well = np.array([g.transport.well_depth for g in gas.species()])
            #sigma
            diam = np.array([g.transport.diameter for g in gas.species()])
            #mu
            dipole = np.array([g.transport.dipole for g in gas.species()])

            #see if molecule is polar
            polar = dipole > 0.0

            #alpha
            polarize = np.array([g.transport.polarizability for g in gas.species()])

            #z_rot
            zrot = np.array([g.transport.rotational_relaxation for g in gas.species()])
            #W_ac
            acentric = np.array([g.transport.acentric_factor for g in gas.species()])

            #determine rotational DOF
            rotDOF = []
            for k in range(ns):
                if gas.species()[k].transport.geometry == 'atom':
                    rotDOF.append(0.0)
                elif gas.species()[k].transport.geometry == 'linear':
                    rotDOF.append(1.0)
                elif gas.species()[k].transport.geometry == 'nonlinear':
                    rotDOF.append(1.5)

            ##########################################
            #Collision Parameters (reduced stuff)
            ##########################################
            r_mass = np.zeros((ns,ns))
            r_well = np.zeros((ns,ns))
            r_diam = np.zeros((ns,ns))
            r_dipole = np.zeros((ns,ns))

            r_deltastar = np.zeros((ns,ns))

            for i in range(ns):
                for j in range(i,ns):
                    # reduced mass
                    r_mass[i,j] = mass[i]*mass[j] / (mass[i]+mass[j])
                    # spheriacl collision diameter
                    r_diam[i,j] = 0.5*(diam[i] + diam[j])
                    # effective well depth
                    r_well[i,j] = np.sqrt(well[i]*well[j])
                    # effective dipole moment
                    r_dipole[i,j] = np.sqrt(dipole[i] * dipole[j])

                    #reduced dipole delta*
                    r_deltastar[i,j] = 0.5 * r_dipole[i,j]**2 / \
                                       ( 4*np.pi*ct.epsilon_0*r_well[i,j]*r_diam[i,j]**3 )

                    #Correct for polarity
                    if polar[i] == polar[j]:
                        f_well = 1.0
                        f_diam = 1.0
                    else:
                        kp,knp = (i,j) if polar[i] else (j,i)
                        d3np = diam[knp]**3
                        d3p = diam[kp]**3
                        alphastar = polarize[knp]/d3np
                        dipolestar = r_dipole[kp,kp]/np.sqrt(4*np.pi*eps0*d3p*well[kp])
                        xi = 1.0+0.25*alphastar*dipolestar**2*np.sqrt(well[kp]/well[knp])
                        f_well = xi**2
                        f_diam = xi**(-1/6)

                    r_well[i,j] *= f_well
                    r_diam[i,j] *= f_diam

                    # properties are symmetric
                    r_mass[j,i] = r_mass[i,j]
                    r_diam[j,i] = r_diam[i,j]
                    r_well[j,i] = r_well[i,j]
                    r_dipole[j,i] = r_dipole[i,j]
                    r_deltastar[j,i] = r_deltastar[i,j]

            ##########################################
            #Viscosities
            ##########################################
            visc = np.zeros((npts,ns))
            ct_visc = np.zeros((npts,ns))
            ct_mix = np.zeros(npts)
            for i,T in enumerate(Ts):
                Tstar = T*kb/well
                omga22 = intrp_o22(Tstar, r_deltastar.diagonal(), grid=False)
                visc[i,:] = (5.0/16.0)*np.sqrt(np.pi*mass*kb*T)/(np.pi*diam**2*omga22)

            ##########################################
            #Thermal Conductivity
            ##########################################
            cond = np.zeros((npts,ns))
            for i,T in enumerate(Ts):
                for k in range(ns):
                    gas.TP = T,101325.0
                    Tstar = kb*298.0/well[k]
                    fz_298 = 1.0 + np.pi**1.5 / np.sqrt(Tstar) * (0.5 + 1.0 / Tstar) + (0.25 * np.pi**2 + 2) / Tstar

                    Tstar = T*kb/well[k]

                    omga22 = intrp_o22(Tstar, r_deltastar[k,k], grid=False)
                    Astar = intrp_Astar(Tstar, r_deltastar[k,k], grid=False)
                    omga11 = omga22/Astar

                    # self diffusion coeff
                    diffcoeff = 3.0/16.0 * np.sqrt(2.0*np.pi/r_mass[k,k])*(kb*T)**1.5 / (np.pi * diam[k]**2 * omga11 )

                    f_int = MW[k]/(Ru*T) * diffcoeff/visc[i,k]
                    cv_rot = rotDOF[k]
                    A_factor = 2.5-f_int
                    fz_tstar = 1.0+np.pi**1.5/np.sqrt(Tstar) * (0.5+1.0/Tstar)+ (0.25*np.pi**2+2)/Tstar
                    B_factor = zrot[k] *fz_298 / fz_tstar + 2.0/np.pi * (5/3 * rotDOF[k] + f_int)
                    c1 = 2.0/np.pi * A_factor/B_factor
                    cv_int = gas.standard_cp_R[k] - 2.5 - cv_rot
                    f_rot = f_int * (1.0+c1)
                    f_trans = 2.5*(1.0-c1*cv_rot/1.5)
                    cond[i,k] = visc[i,k]/MW[k]*Ru*(f_trans*1.5 + f_rot*cv_rot + f_int*cv_int)

            ##########################################
            # Binary Diffusion
            ##########################################
            diff = np.zeros((npts,ns,ns))
            for k in range(ns):
                for j in range(k,ns):
                    for i,T in enumerate(Ts):
                        Tstar = T*kb/r_well[j,k]

                        omga22 = intrp_o22(Tstar, r_deltastar[j,k], grid=False)
                        Astar = intrp_Astar(Tstar, r_deltastar[j,k], grid=False)
                        omga11 = omga22/Astar

                        # To get pressure dependence, we evaluate the coeff at unit pressure
                        # then when we actually NEED the coeff, we use divide by the real pressure
                        diffcoeff = 3.0/16.0 * np.sqrt(2.0*np.pi/r_mass[k,j])*(kb*T)**1.5 / ( np.pi * r_diam[j,k]**2 * omga11 )

                        diff[i,k,j] = diffcoeff
                        diff[i,j,k] = diff[i,k,j]


            # Create and set the polynoial coefficients
            logTs = np.log(Ts)
            sqrtTs = np.sqrt(Ts)

            #We fit the visc pol'y to the sqrtT as visc is proportional to sqrtT
            visc = visc/sqrtTs[:,None]
            w = 1.0/(visc**2)
            self.mu_poly    = [list(np.polyfit(logTs,visc[:,k],deg=deg,w=w[:,k])) for k in range(ns)]

            #We fit the cond pol'y to the sqrtT as cond is proportional to sqrtT
            cond = cond/np.sqrt(Ts[:,None])
            w = 1.0/(cond**2)
            self.kappa_poly = [list(np.polyfit(logTs,cond[:,k],deg=deg,w=w[:,k])) for k in range(ns)]

            Dij = []
            diff = diff/Ts[:,None,None]**1.5
            w = 1.0/(diff**2)
            for k in range(ns):
                for j in range(k,ns):
                    Dij.append(list(np.polyfit(logTs,diff[:,k,j],deg=deg,w=w[:,k,j])))

            self.Dij_poly = Dij


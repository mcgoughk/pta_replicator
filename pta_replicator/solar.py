#get the pulsar data
#get the angle between the sun and earth
#get the angle between the earth and the pulsar
#calculate the solar DM 

#get pulsar
#get pulsar theta and phi
#guess ~1kpc distance
#sun location
#solarsystem barycenter theta and phi
#compute angles
#calculate the solar wind contribution

import numpy as np
import inspect
import jax.numpy as jnp
import warnings
from scipy import interpolate
from pta_replicator.simulate import SimulatedPulsar
from pta_replicator.constants import DAY_IN_SEC, YEAR_IN_SEC


from . import const
from . import matrix

AU_light_sec = const.AU / const.c  # 1 AU in light seconds
AU_pc = const.AU / const.pc        # 1 AU in parsecs (for DM normalization)

def theta_impact(psr): #calcuates the impact of the angle between the sun and pulsar for DM
    """From enterprise_extensions: use the attributes of an Enterprise
    Pulsar object to calculate the solar impact angle.
    Returns solar impact angle (rad), distance to Earth (R_earth),
    impact distance (b), perpendicular distance (z_earth)."""

    earth = psr.planetssb[:, 2, :3]
    sun = psr.sunssb[:, :3]
    earthsun = earth - sun

    R_earth = np.sqrt(np.einsum('ij,ij->i', earthsun, earthsun))
    Re_cos_theta_impact = np.einsum('ij,ij->i', earthsun, psr.pos_t)

    theta_impact = np.arccos(-Re_cos_theta_impact / R_earth)
    b = np.sqrt(R_earth**2 - Re_cos_theta_impact**2)

    return theta_impact, R_earth, b, -Re_cos_theta_impact

def make_solardm(psr):
    theta, r_earth, _, _ = theta_impact(psr)
    shape = matrix.jnparray(AU_light_sec * AU_pc / r_earth / np.sinc(1 - theta/np.pi) * 4.148808e3 / psr.freqs**2)

    def solardm(n_earth): # calculates the DM around earth
        return n_earth * shape

def add_solar_noise(psr: SimulatedPulsar, theta_impact):
    return

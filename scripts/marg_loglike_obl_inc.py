import warnings
warnings.filterwarnings("ignore")

import starry
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import pymc3.math as pmm
import pymc3_ext as pmx
import exoplanet
from starry_process import StarryProcess, MCMCInterface
from starry_process.math import cho_factor, cho_solve
import starry_process 
import theano
theano.config.gcc__cxxflags += " -fexceptions"
theano.config.on_opt_error = "raise"
theano.tensor.opt.constant_folding
theano.graph.opt.EquilibriumOptimizer
import aesara_theano_fallback.tensor as tt
from theano.tensor.slinalg import cholesky
from corner import corner
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
from tqdm import tqdm
from theano.tensor.random.utils import RandomStream
import scipy.linalg as sl
import scipy.stats as ss
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

starry.config.quiet = True

from starry_starry_process import StarryStarryProcess

true_obl = np.random.uniform(-180, 180)
true_inc = np.random.uniform(0, 180)
orig = True

truths = {"planet.inc": 90,
          "planet.r": 0.1,
          "planet.t0": 0.5,
          "planet.porb": 1.,
          "star.inc": true_inc,
          "star.obl": true_obl,
          "star.prot": 5.,
          "sp.a": 0.27,
          "sp.b": 0.14,
          "sp.c": 0.23,
          "sp.r": 27,
          "sp.n": 2.1,
          "u": [0.4, 0.2]}

map = starry.Map(15)
map.inc = truths['star.inc']
map.obl = truths['star.obl']

star = starry.Primary(map, r=1., m=1., prot=truths['star.prot']) 
planet = starry.Secondary(
    starry.Map(0,0),
    porb=truths['planet.porb'],
    t0=truths['planet.t0'],
    r=truths['planet.r'],
)

planet.inc = truths['planet.inc']

sp = StarryProcess(
        a = truths["sp.a"],
        b = truths["sp.b"],
        c = truths["sp.c"],
        r = truths["sp.r"],
        n = truths["sp.n"],
    )
y_true = sp.sample_ylm().eval().reshape(-1)
y_true[0] += 1

map[:,:] = y_true

sys = starry.System(star, planet)
t = np.arange(0, 5, 6 / 24 / 60)

map.show()


def gp_model(t, flux_obs, sigma_flux, original=orig):
    starry.config.lazy = True
        
    with pm.Model() as model:

        if original:
            stellar_ori_x = pm.Normal('stellar_ori_x', mu=0, sigma=1, testval=1)
            stellar_ori_y = pm.Normal('stellar_ori_y', mu=0, sigma=1, testval=1)
            stellar_ori_z = pm.Normal('stellar_ori_z', mu=0, sigma=1, testval=1)

        else:
            stellar_ori_x_ = pm.Normal('stellar_ori_x_', mu=0, sigma=1, testval=1)
            stellar_ori_y_ = pm.Normal('stellar_ori_y_', mu=0, sigma=1, testval=1)
            stellar_ori_z_ = pm.HalfNormal('stellar_ori_z_', sigma=1, testval=1)

            stellar_ori_x = stellar_ori_x_
            stellar_ori_y = (tt.sqrt(2)/2) * (stellar_ori_z_ + stellar_ori_y_)
            stellar_ori_z = (tt.sqrt(2)/2) * (stellar_ori_z_ - stellar_ori_y_)

        stellar_obl = pm.Deterministic('stellar_obl', 180.0/np.pi*tt.arctan2(stellar_ori_y, stellar_ori_x))
        stellar_inc = pm.Deterministic('stellar_inc', 180.0/np.pi*tt.arccos(stellar_ori_z / tt.sqrt(tt.square(stellar_ori_x) + tt.square(stellar_ori_y) + tt.square(stellar_ori_z))))

        map_model = starry.Map(15)
        map_model.inc = stellar_inc 
        map_model.obl = stellar_obl 
        star_model = starry.Primary(map_model, r=1., m=1., prot=truths['star.prot']) 
        planet_model = starry.Secondary(
            starry.Map(0,0),
            porb=truths['planet.porb'],
            t0=truths['planet.t0'],
            r=truths['planet.r'],
        )

        planet.inc = truths['planet.inc']

        map_model[:,:] = y_true

        sys_model = starry.System(star_model, planet_model)
        pm.Deterministic('design_matrix', sys_model.design_matrix(t)[:,:-1])
        sp_model = StarryProcess(
            a = truths["sp.a"],
            b = truths["sp.b"],
            c = truths["sp.c"],
            r = truths["sp.r"],
            n = truths["sp.n"],
            ydeg = 15,
        )
        ssp_model = StarryStarryProcess(sys_model, sp_model)
        
        pm.Potential('marginal_likelihood', ssp_model.marginal_likelihood(t, flux_obs, sigma_flux=sigma_flux))

    return model


def get_design_matrix(original=orig):
    model = gp_model(t, np.zeros_like(t), np.ones_like(t))
    if original:
        true_pt = {
        model.stellar_ori_x: np.sin(truths['star.inc'] * np.pi / 180) * np.cos(truths['star.obl'] * np.pi / 180),
        model.stellar_ori_y: np.sin(truths['star.inc'] * np.pi / 180) * np.sin(truths['star.obl'] * np.pi / 180),
        model.stellar_ori_z: np.cos(truths['star.inc'] * np.pi / 180)
    }
    else:
        x = np.sin(truths['star.inc'] * np.pi / 180) * np.cos(truths['star.obl'] * np.pi / 180)
        y = np.sin(truths['star.inc'] * np.pi / 180) * np.sin(truths['star.obl'] * np.pi / 180)
        z = np.cos(truths['star.inc'] * np.pi / 180)

        true_pt = {
            model.stellar_ori_x_: x,
            model.stellar_ori_y_: (np.sqrt(2)/2) * (y - z),
            model.stellar_ori_z_: (np.sqrt(2)/2) * (y + z)
        }
    
    model_dm = model.design_matrix.eval(true_pt)

    return model_dm

model_dm = get_design_matrix()
rng = np.random.default_rng(9878997)
flux_true = model_dm @ y_true
sigma_flux = 1e-3*np.ones_like(flux_true)
flux_obs = flux_true + sigma_flux*rng.normal(size=len(t))

model = gp_model(t, flux_obs, sigma_flux)

def marginal_likelihood(obl, inc, original=orig):
    if original:
        pt = {
            'stellar_ori_x': np.sin(inc * np.pi / 180) * np.cos(obl * np.pi / 180),
            'stellar_ori_y': np.sin(inc * np.pi / 180) * np.sin(obl * np.pi / 180),
            'stellar_ori_z': np.cos(inc * np.pi / 180)
        }
    else:
        x = np.sin(inc * np.pi / 180) * np.cos(obl * np.pi / 180)
        y = np.sin(inc * np.pi / 180) * np.sin(obl * np.pi / 180)
        z = np.cos(inc * np.pi / 180)

        pt = {
            'stellar_ori_x_': x,
            'stellar_ori_y_': (np.sqrt(2)/2) * (y - z),
            'stellar_ori_z__log__': np.log(abs((np.sqrt(2)/2) * (y + z)))
        }

    return model.logp(pt)

print(f'obliquity: {true_obl}, inclination: {true_inc}')
print(f"the marginal likelihood for obliquity {true_obl} and inclination {true_inc} is {marginal_likelihood(true_obl, true_inc)}")
print(f"the marginal likelihood for obliquity {-true_obl} and inclination {180-true_inc} is {marginal_likelihood(-true_obl, 180-true_inc)}")


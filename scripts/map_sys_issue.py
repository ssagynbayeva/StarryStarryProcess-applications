import starry
import matplotlib.pyplot as plt
import numpy as np

starry.config.quiet = True

true_obl = -30 

truths = {"planet.inc": 90,
          "planet.r": 0.1,
          "planet.t0": 0.5,
          "planet.porb": 1.,
          "star.inc": 90,
          "star.obl": true_obl,
          "star.prot": 5.,
          "sp.mu": 30,
          "sp.sigma": 1.,
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

# map.spot(contrast=0.9, radius=30, lon=40, lat=30)
# map.spot(contrast=0.9, radius=20, lon=-10, lat=-30)

map.show()

sys = starry.System(star, planet)
t = np.arange(0, 5, 6 / 24 / 60)

X = sys.design_matrix(t).eval()

theta = (360 * t / truths["star.prot"]) % 360
xo, yo, zo = sys.position(t)
xo = xo.eval()[1]
yo = yo.eval()[1]
zo = zo.eval()[1]
A = map.design_matrix(
    theta=theta, xo=xo, yo=yo, zo=zo, ro=truths["planet.r"]
).eval()

plt.plot(X[:,:-1]-A)
plt.title('design matrix difference')
plt.show()


plt.plot(t, map.flux(theta=theta, xo=xo, yo=yo, zo=zo, ro=truths["planet.r"]).eval(), label='map.flux')
plt.plot(t, sys.flux(t, total=False)[0].eval(), label='sys.flux')
plt.legend()
# plt.xlim(0.4,0.6)
plt.show()

a = (1*np.square(truths['planet.porb']/365.25))**(1/3) * 215.03 # Solar radii
b = 0
tdur = truths['planet.porb'] / np.pi * np.arcsin(np.sqrt((truths['planet.r']+1)**2 - b**2) / a)

print(tdur)

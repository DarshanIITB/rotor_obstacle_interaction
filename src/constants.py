from math import *
import numpy as np

class ModelParams:
    inversion_tolerance = 1e-12
    farfield_factor = 10.0
    trailing_edge_wake_shed_factor = 0.25
    unsteady_problem = False
    static_wake_length = 1.0
    static_wake = False
    use_vortex_core_model = False

N = 24  # number of divisions in blade span also in one revolution
NR = 7
alpha_d = 0  # assuming Tip path plane
alpha = alpha_d * (pi / 180)
N_count = 40
Vv = 0  # tip path plane in radians
u_inf = 0  # this is Y direction
v_inf = 0  # this is X direction
w_inf = 0 + u_inf * sin(alpha) - Vv
V_vec = [v_inf, u_inf, w_inf]
IM = 7 + 1
a = 5.75  # lift curve slope per radian from literature
Cdo = 0.0113  # co-efficient of drag from literature
b = 2
Number_of_rotors = 1
Blades_per_turbine = b / Number_of_rotors
blades = b  # number of blades1


# tip path plane in radians

RPM = 383  # rpm of rotor at criuse
omega = (
    2 * pi * RPM
) / 60  # angular velocity                                                                  #increment of azimuthal angle
rho = 1.225  # density of air
chord_hub = 0.325  # chord at root cut out                                                              #tip radius

R_tip = 5.5
R_hub = 0.1 * R_tip
Axial_diplacement = 2 * R_tip
span = R_tip - R_hub

dr = span / (IM - 1)
radius = np.linspace(R_hub, R_tip, IM)
D = 2 * R_tip  # blade diameter

taper_ratio = 1  # taper ratio
chord_tip = chord_hub * taper_ratio  # chord at blade tip
taper_slope = (chord_tip - chord_hub) / (R_tip - R_hub)  # term T
chord_root = chord_hub - taper_slope * R_hub  # chord at root

chord = chord_root + taper_slope * radius

theta_tip = 2  # twist at tip of blade
theta_hub = 12  # twist atroot cut out
F = (theta_tip - theta_hub) / (R_tip - R_hub)  # term F
E = theta_hub - F * R_hub  # term E
mu = (u_inf * cos(alpha)) / (omega * R_tip)  # advance ratio

lc = Vv / (omega * R_tip)

# print("Theoretical thrust(N):",T)
blade_twist = (theta_tip - theta_hub) * (pi / 180)  # radians
theta = (E + F * radius) * (pi / 180)  #
solidity = (b * chord) / (pi * R_tip)

dt = (60 / RPM) * (1 / N)  # dpsi

rad_sec = (2 * pi * RPM) / 60
dpsi_d = 360 / N

dpsi = dpsi_d * (pi / 180)
NT = (NR * (360)) / dpsi_d
T = NT * dt
JM = int(NT)
beta = 0

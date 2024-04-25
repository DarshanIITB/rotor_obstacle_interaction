import numba
import time
import numpy as np
import sympy as sp  # importing numpy library as sp
from scipy.integrate import simps
from math import *
import matplotlib.pyplot as plt

from utils import Rotate, Biot_Savarts_Law2, Biot_Savarts_Perfromance, calc_VIND_obj
from surface import Surface
from plot3d import PLOT3D
from domain import Domain
from wake import Wake
from constants import ModelParams
from solver import Solver

from constants import *

ModelParams.unsteady_problem = False
ModelParams.static_wake = True
ModelParams.static_wake_length = 10.0
surface = Surface()

mesh = PLOT3D()
filename = "apame.x"
mesh.set_surf(surface)
mesh.read_surface(filename)

print(f"Mesh has {len(mesh.surf.nodes)} nodes.")
print(f"Mesh has {len(mesh.surf.panels)} panels.")
print(f"Panel neighbours: {len(mesh.surf.panel_neighbours)}")

surface.rotate_surface(np.array([0., 0.0, 0.0]), "deg")
surface.translate_surface(np.array([-1., 0., -10.]))

mesh.plot_geometry()

time_step = 0.5
fluid_density = 1.225

free_stream_vel = np.array([0, 0, 0])
wake = Wake()
wake.add_lifting_surf(surface)
wake.init(free_stream_vel, time_step)
solver = Solver()
solver.add_surf(surface)
solver.add_wake(wake)
solver.set_fluid_density(fluid_density)
solver.set_free_stream_vel(free_stream_vel)
solver.set_ref_vel(free_stream_vel)

Start_time = time.time()

L_ig = sp.Symbol("lambda_ig")  # declaring symbol lambda_ig
L_i = sp.Symbol("lambda_i")

# %%
# N = 24  # number of divisions in blade span also in one revolution
# NR = 7
# alpha_d = 0  # assuming Tip path plane
# alpha = alpha_d * (pi / 180)
# N_count = 100
# Vv = 0  # tip path plane in radians
# u_inf = 0  # this is Y direction
# v_inf = 0  # this is X direction
# w_inf = 0 + u_inf * sin(alpha) - Vv
# V_vec = [v_inf, u_inf, w_inf]
# IM = 3 + 1
# a = 5.75  # lift curve slope per radian from literature
# Cdo = 0.0113  # co-efficient of drag from literature
# b = 2
# Number_of_rotors = 1
# Blades_per_turbine = b / Number_of_rotors
# blades = b  # number of blades1


# # tip path plane in radians

# RPM = 383  # rpm of rotor at criuse
# omega = (
#     2 * pi * RPM
# ) / 60  # angular velocity                                                                  #increment of azimuthal angle
# rho = 1.225  # density of air
# chord_hub = 0.325  # chord at root cut out                                                              #tip radius

# R_tip = 5.5
# R_hub = 0.1 * R_tip
# Axial_diplacement = 2 * R_tip
# span = R_tip - R_hub

# dr = span / (IM - 1)
# radius = np.linspace(R_hub, R_tip, IM)
# D = 2 * R_tip  # blade diameter

# taper_ratio = 1  # taper ratio
# chord_tip = chord_hub * taper_ratio  # chord at blade tip
# taper_slope = (chord_tip - chord_hub) / (R_tip - R_hub)  # term T
# chord_root = chord_hub - taper_slope * R_hub  # chord at root

# chord = chord_root + taper_slope * radius

# theta_tip = 2  # twist at tip of blade
# theta_hub = 12  # twist atroot cut out
# F = (theta_tip - theta_hub) / (R_tip - R_hub)  # term F
# E = theta_hub - F * R_hub  # term E
# mu = (u_inf * cos(alpha)) / (omega * R_tip)  # advance ratio

# lc = Vv / (omega * R_tip)

# # print("Theoretical thrust(N):",T)
# blade_twist = (theta_tip - theta_hub) * (pi / 180)  # radians
# theta = (E + F * radius) * (pi / 180)  #
# solidity = (b * chord) / (pi * R_tip)

# dt = (60 / RPM) * (1 / N)  # dpsi

# rad_sec = (2 * pi * RPM) / 60
# dpsi_d = 360 / N

# dpsi = dpsi_d * (pi / 180)
# NT = (NR * (360)) / dpsi_d
# T = NT * dt
# JM = int(NT)
# print("Coefficient of thrust ", CT)
print(
    "JM is",
    JM,
    "  IM is",
    IM,
    ",dpsi:",
    round(dpsi * 180 / pi, 2),
    "twist",
    blade_twist * (180 / pi),
    "NR ",
    NR,
)
print("Advance ratio:", mu)
print("Tip Speed (m/s):", omega * R_tip)
# beta = 0

wake_age_array = np.linspace(0, JM * dpsi * 180 / pi, JM)
V_INF = np.zeros((JM, 3))
for i in range(3):
    for j in range(JM):
        V_INF[j, i] = V_vec[i]

psi_array = np.linspace(0, 2 * pi, N)

# %%


# PTLM
location = R_hub  # location of blade root cut out
PTC_inflow_array = (
    []
)  # array to store inflow ratio after solving Prantl's tip loss model
location_array = []
for i in range(IM):  # loop for elemental radial location
    dif = 1  # to start loop a initail diffferece is provided
    lmd_1 = 1  # initial guess values for inflow ratio
    while dif > 0.001:  # to iterate till the difference is in order of 0.0001
        f = (
            0.5 * b * (1 - (radius[i] / R_tip)) / lmd_1
        )  # f term  of  Prantl's tip loss model
        if f < 0.00001:
            FL = (2 / pi) * acos(e ** (-1 * 0.00001))
        else:
            FL = (2 / pi) * acos(e ** (-1 * f))  # F term of  Prantl's tip loss model
        lmd_2 = np.sqrt(
            (((a * solidity[i]) / (16 * FL)) - 0.5 * lc) ** (2)
            + ((solidity[i] * a * theta[i] * radius[i]) / (FL * 8 * R_tip))
        ) - (((a * solidity[i]) / (16 * FL)) - 0.5 * lc)
        dif = abs(lmd_1 - lmd_2)  # checking difference
        lmd_1 = lmd_2  # assigning new inflow ratio value
    PTC_inflow_array.append(lmd_1)  # filling up the  PTC_inflow_array array
# plt.plot(PTC_inflow_array)

phi_array = np.arctan(PTC_inflow_array * (R_tip / radius))
alpha_array = theta - phi_array
Cd_array = Cdo + 1.25 * (alpha_array) ** 2
Cl_array = a * (alpha_array)
# UT= omega * radius + u_inf*cos(alpha)*sin(psi_array[j])
circulation = (
    0.5 * (omega * radius + u_inf * cos(alpha) * sin(pi / 2)) * chord * Cl_array
)
circulation_old = circulation
plt.figure(1, dpi=120)
plt.title("Circulation")
plt.plot(radius, circulation_old, "b.-")
plt.show()
gamma_max = max(circulation)

# %%
# Thrust2 CALCULATION and power calculation
Thrust2_array = np.zeros(IM)
Power2_array = np.zeros(IM)
UP_array = np.zeros(IM)
T_a = 0.5 * rho * b  # constant part in thrust equation
for i in range(IM):
    Ut = omega * radius[i]  # tangential velocity
    Up = PTC_inflow_array[i] * omega * R_tip  # perpendicular velocity
    Cd = Cdo + 1.25 * (theta[i] - atan(Up / Ut)) ** 2  # drag equation in terms of r
    Cl = a * (theta[i] - atan(Up / Ut))  # lift equation   in terms of r
    dT = (
        0.5
        * rho
        * b
        * (Up**2 + Ut**2)
        * chord[i]
        * (Cl * cos(atan(Up / Ut)) - Cd * sin(atan(Up / Ut)))
    )
    dP = (
        0.5
        * rho
        * b
        * omega
        * radius[i]
        * (Up**2 + Ut**2)
        * chord[i]
        * (Cd * cos(atan(Up / Ut)) + Cl * sin(atan(Up / Ut)))
    )
    Thrust2_array[i] = dT
    Power2_array[i] = dP
    UP_array[i] = Up

Thrust = np.trapz(Thrust2_array, dx=dr) / cos(alpha)
Power = np.trapz(Power2_array, dx=dr) / cos(alpha)
CT = Thrust / (rho * pi * R_tip**2 * (omega * R_tip) ** 2)  # Thrust co efficient #""";
CP = Power / (rho * pi * R_tip**2 * (omega * R_tip) ** 3)  # Thrust co efficient #""";
# %%
for i in reversed(range(IM)):
    # print(i,location_array[i])
    if circulation[i] > circulation[i - 1]:
        location_cutoff = radius[i]
        # print(" Circulation Cutoff location:",location_cutoff )
        break
    else:
        pass


CT = Thrust / (rho * pi * R_tip**2 * (omega * R_tip) ** 2)
LHS = CT
RHS = 2 * L_ig * sp.sqrt(mu**2 + (mu * tan(alpha) + L_ig) ** 2)
function = LHS - RHS
diff_function = function.diff(L_ig)  # differentiation of equation
h1 = sp.lambdify(L_ig, function)  # defines value of L_ig in  equation
h2 = sp.lambdify(
    L_ig, diff_function
)  # defines value of L_ig in differentiated  equation
i = 0.01  # initial guess for  Newton-Raphson method
for j in range(100):  # Newton-Raphson method
    h = -1 * h1(i) / h2(i)  # h= -f(x)/df(x)
    i = i + h  # Xi= X(i-1)+h
L_ig_calculated = i  # calculated value of L_ig
LG = L_ig_calculated + L_i
# Thrust2 CALCULATION and power calculation
#  below mentioned are the arrays to store the velues of function to calculate T, P,Q ,Overall rolling and pitching moments at each azimuthal location
dOPM_array = np.zeros(N)
dORM_array = np.zeros(N)

dT_blade_array = np.zeros(N)
dOQi_array = np.zeros(N)
dOQo_array = np.zeros(N)
dT_array = np.zeros(N)
dOPi_array = np.zeros(N)
dOPo_array = np.zeros(N)
dbeta_dt = 0  # TPP as a reference frame
beta_o_d = 0  # TPP frame of reference # beta   in degrees
beta_1s_d = 0  # beta_1s kept 0 or as pilot input in degrees
beta_1c_d = 0  # beta_1c kept 0 pilot input  degrees
theta_1s = (0) * (pi / 180)  # theta_1s  in degrees at trim conditio
theta_1c = (0) * (pi / 180)  # theta_1c  in degrees at trim conditio
for j in range(N):

    beta = (
        beta_o_d + beta_1c_d * cos(psi_array[j]) + beta_1s_d * sin(psi_array[j])
    ) * (
        pi / 180
    )  # beta

    # arrays to store sectional Lift,  induced & profile Torque ,induced & profile Power at each radial location
    dL_array = np.zeros(IM)
    dP_array = np.zeros(IM)
    dR_array = np.zeros(IM)
    dQi_array = np.zeros(IM)
    dQo_array = np.zeros(IM)
    dPi_array = np.zeros(IM)
    dPo_array = np.zeros(IM)
    for i in range(IM):  # loop to solve the dL,dQi,dQo,dPi,dPo, dPM, dRM

        # newton Raphson method to solve inflow ratio at each radial location
        LHS = (L_i) / (L_ig_calculated)  # LHS
        RHS = 1 + ((1.333 * mu) / (1.2 * (L_i + L_ig_calculated) + mu)) * (
            (radius[i] * cos(psi_array[j])) / R_tip
        )  # RHS
        function = LHS - RHS  # funtion equated to 0
        diff_function = function.diff(L_i)  # differentiation of equation
        h1 = sp.lambdify(L_i, function)  # defines value of L_i in  equation
        h2 = sp.lambdify(
            L_i, diff_function
        )  # defines value of L_i  in differentiated  equation
        inr = 0
        difference = 1  # initial guess for  Newton-Raphson method
        while difference > 0.001:
            h = -1 * h1(inr) / h2(inr)  # h= -f(x)/df(x)
            difference = abs(h)
            inr = inr + h  # Xi= X(i-1)+h
        L_i_calculated = (
            inr  # calculated value of L_i (inflow ratio) at given radial location
        )

        theta_equation = (
            theta[i] + theta_1c * cos(psi_array[j]) + theta_1s * sin(psi_array[j])
        )  # theta in tersm of r i.e. theta_twd=E+Fr

        Up = (
            u_inf * sin(alpha)
            + (omega * R_tip * (L_i_calculated))
            + radius[i] * dbeta_dt
            + u_inf * sin(beta) * cos(psi_array[j])
        )  # Up
        Ut = omega * radius[i] + u_inf * cos(alpha) * sin(psi_array[j])

        dL_array[i] = (
            0.5 * rho * chord[i] * a * (theta_equation * Ut**2 - Ut * Up)
        )  # filling the  corresponding array
        dP_array[i] = (
            dL_array[i] * radius[i] * cos(psi_array[j])
        )  # filling the  corresponding array

        dR_array[i] = (
            dL_array[i] * radius[i] * sin(psi_array[j])
        )  # filling the  corresponding array

        dDi = (
            1.5 * 0.5 * rho * chord[i] * a * (theta_equation - atan(Up / Ut)) * Up * Ut
        )  # Di term in induced torque equation
        dQi_array[i] = dDi * radius[i]  # filling the  corresponding array

        dDo = (
            0.5 * rho * Ut * Ut * chord[i] * (Cdo)
        )  # Do term in  profile torque equation
        dQo_array[i] = dDo * radius[i]  # filling the  corresponding array

        dPi_array[i] = dDi * Ut  # filling the  corresponding array
        dPo_array[i] = dDo * Ut  # filling the  corresponding array

    dT_array[j] = np.trapz(
        dL_array, radius
    )  # thust produced at given azimuthal location

    dOPM_array[j] = np.trapz(
        dP_array, radius
    )  # pitching moment at given azimuthal location

    dORM_array[j] = np.trapz(
        dR_array, radius
    )  # rolling moment at given azimuthal location

    dOQi_array[j] = np.trapz(
        dQi_array, radius
    )  # induced torque at given azimuthal location                                              #filling in  corresponding array
    dOQo_array[j] = np.trapz(
        dQo_array, radius
    )  # profile torque at given azimuthal location

    dOPi_array[j] = np.trapz(
        dPi_array, radius
    )  # induced power at given azimuthal location
    dOPo_array[j] = np.trapz(
        dPo_array, radius
    )  # profile power at given azimuthal location
    # print("\u03A8 :",float32(psi_array[j]*(180/pi)),"  dT:",dT_array[j],"  dPM:",dOPM_array[j],"   dRM:",dORM_array[j])

THRUST = np.trapz(dT_array, psi_array) * (
    b / (2 * pi)
)  # thust produced at given azimuthal location

OVERALL_PITCHING_MOMENT = np.trapz(dOPM_array, psi_array) * (
    b / (2 * pi)
)  # Overall Pitching Moment produced by rotor
OVERALL_ROLLING_MOMENT = np.trapz(dORM_array, psi_array) * (
    b / (2 * pi)
)  # Overall Rolling Moment produced by rotor
INDUCED_TORQUE = np.trapz(dOQi_array, psi_array) * (
    b / (2 * pi)
)  # Induced Torque produced by rotor
PROFILE_TORQUE = np.trapz(dOQo_array, psi_array) * (
    b / (2 * pi)
)  # Profile Torque produced by rotor

TOTAL_TORQUE = INDUCED_TORQUE + PROFILE_TORQUE  # Total Torque produced by rotor

INDUCED_POWER = np.trapz(dOPi_array, psi_array) * (
    b / (2 * pi)
)  # Induced Power produced by rotor
PROFILE_POWER = np.trapz(dOPo_array, psi_array) * (
    b / (2 * pi)
)  # Profile Power produced by rotor
TOTAL_POWER = abs(INDUCED_POWER) + PROFILE_POWER  # Total Power produced by rotor

# figure(1)
# xlabel("Azimuthal location in degrees")
# ylabel("Thrust produced by a single blade")
# plot(psi_array,dT_blade_array,'b-')
print(">>>>>>>>>>>>  RESULTS  <<<<<<<<<<<<")
print("Thrust at Trim condition (N):", round(THRUST, 3))
print("Overall Pitching Moment (Nm):", round(OVERALL_PITCHING_MOMENT, 3))
print("Overall Rolling Moment (Nm):", round(OVERALL_ROLLING_MOMENT, 3))
print("Induced Torque (Nm):", round(INDUCED_TORQUE, 3))
print("Profile Torque (Nm):", round(PROFILE_TORQUE, 3))
print("Total Torque (Nm):", round(TOTAL_TORQUE, 3))
print("Induced Power (kW):", abs(INDUCED_POWER * 0.001))
print("Profile Power (kW):", round(PROFILE_POWER * 0.001, 3))
print("Total Power (kW):", round(TOTAL_POWER * 0.001, 3))

CT = THRUST / (rho * pi * R_tip**2 * (omega * R_tip) ** 2)  # Thrust co efficient #""";
CP = TOTAL_POWER / (
    rho * pi * R_tip**2 * (omega * R_tip) ** 3
)  # Thrust co efficient #""";
print("ThrustA2: {} N and CT: {}".format(round(Thrust, 3), CT))
print("PowerA2: {} kW and CP: {}".format(round(Power / 1000, 3), CP))
print("Thrust: {} N and CT: {}".format(round(THRUST, 3), CT))
print("Power: {} kW and CP: {}".format(round(TOTAL_POWER / 1000, 3), CP))


# plt.plot(UP_array)

# %%


def HoverWake(CT_input, mu_in):
    CT = CT_input
    A = 0.78
    lam = 0.145 + 27 * CT
    Zv_array = np.zeros((JM, IM))
    array_X1 = np.zeros((JM, IM, b))
    array_Y1 = np.zeros((JM, IM, b))
    array_Z1 = np.zeros((JM, IM, b))

    array_Xn = np.zeros(JM)
    array_Yn = np.zeros(JM)
    array_Zn = np.zeros(JM)
    for i in range(IM):
        if i >= 0 and i < (IM - 1):  # trailing vortex sheet
            for j in range(JM):  # TIME LOOP
                psi = j * dpsi
                R_vortex = radius[i] * (A + (1 - A) * exp(-lam * psi))
                chord_value = chord_root + taper_slope * radius[i]
                solidity_value = (b * chord_value) / (pi * R_tip)
                K1 = -2.2 * sqrt(0.5 * CT)
                K2 = -2.7 * sqrt(0.5 * CT)

                if psi <= ((2 * pi) / b):
                    Zv_array[j][i] = radius[i] * (K1 * psi)
                else:
                    Zv_array[j][i] = radius[i] * (
                        K1 * ((2 * pi) / b) + K2 * (psi - ((2 * pi) / b))
                    )

                for bn in range(1, b + 1):
                    if bn <= Blades_per_turbine:
                        array_X1[j][i][bn - 1], array_Y1[j][i][bn - 1] = Rotate(
                            R_vortex, 0, psi + ((bn - 1) / Blades_per_turbine) * 2 * pi
                        )
                        array_Z1[:, :, bn - 1] = Zv_array
                    else:
                        array_X1[j][i][bn - 1], array_Y1[j][i][bn - 1] = Rotate(
                            R_vortex, 0, psi + ((bn - 1) / Blades_per_turbine) * 2 * pi
                        )
                        array_Z1[:, :, bn - 1] = Zv_array - Axial_diplacement

        if i == (IM - 1):  # Tip vortex
            for j in range(JM):  # TIME LOOP
                psi = j * dpsi
                R_vortex = radius[i] * (A + (1 - A) * exp(-lam * psi))
                chord_value = chord_root + taper_slope * radius[i]
                solidity_value = (b * chord_value) / (pi * R_tip)

                K1 = -0.25 * ((CT / solidity_value) + 0.001 * blade_twist * (180 / pi))

                K2 = -(1.41 + 0.0141 * blade_twist * (180 / pi)) * sqrt(0.5 * CT)

                if psi <= ((2 * pi) / b):
                    Zv_array[j][i] = radius[i] * (K1 * psi)
                else:
                    Zv_array[j][i] = radius[i] * (
                        K1 * ((2 * pi) / b) + K2 * (psi - ((2 * pi) / b))
                    )

                for bn in range(1, b + 1):
                    if bn <= Blades_per_turbine:
                        array_X1[j][i][bn - 1], array_Y1[j][i][bn - 1] = Rotate(
                            R_vortex, 0, psi + ((bn - 1) / Blades_per_turbine) * 2 * pi
                        )
                        array_Z1[:, :, bn - 1] = Zv_array
                    else:
                        array_X1[j][i][bn - 1], array_Y1[j][i][bn - 1] = Rotate(
                            R_vortex, 0, psi + ((bn - 1) / Blades_per_turbine) * 2 * pi
                        )
                        array_Z1[:, :, bn - 1] = Zv_array - Axial_diplacement

    return array_Y1, array_X1, array_Z1  #########


# %%


array_Xprs, array_Yprs, array_Zprs = HoverWake(CT, mu)

Positions_compare = np.zeros((JM, IM, 3, b))
for j in range(JM):
    for i in range(IM):
        Positions_compare[j, i, 0, 0], Positions_compare[j, i, 1, 0] = Rotate(
            array_Xprs[j, i, 0], array_Yprs[j, i, 0], -1 * dpsi
        )
        Positions_compare[j, i, 2, 0] = array_Zprs[j, i, 0]

array_Xprsnr, array_Yprsnr, array_Zprsnr = (
    np.divide(array_Xprs, R_tip),
    np.divide(array_Yprs, R_tip),
    np.divide(array_Zprs, R_tip),
)
for bn in range(1, b + 1):
    print((180 / pi) * ((bn - 1) / Blades_per_turbine) * 2 * pi)

plt.figure(3, figsize=(5, 5), dpi=150)
ax = plt.axes(projection="3d")
array_Xprs, array_Yprs, array_Zprs = HoverWake(CT, mu)
Positions_compare = np.zeros((JM, IM, 3, b))
array_Xprsnr, array_Yprsnr, array_Zprsnr = (
    np.divide(array_Xprs, R_tip),
    np.divide(array_Yprs, R_tip),
    np.divide(array_Zprs, R_tip),
)
ax.set_xlabel("Y-axis")
ax.set_ylabel("X-axis")
ax.set_zlabel("Z-axis")
# ax.view_init(0,0)
for bl in range(0, b):
    # for i in range(IM-1):
    #   plt.plot(array_Xprsnr[:,i,bl],array_Yprsnr[:,i,bl],array_Zprsnr[:,i,bl],color='blue',linewidth=1)
    plt.plot(
        array_Xprsnr[:, IM - 1, bl],
        array_Yprsnr[:, IM - 1, bl],
        array_Zprsnr[:, IM - 1, bl],
        color="red",
        linewidth=1,
    )
plt.show()


# %%
# @numba.njit
def EX2BMethod():
    Positions_OLD = np.zeros((JM, IM, 3, b))

    Positions_NEW = np.zeros((JM, IM, 3, b))

    Blade_positions = np.zeros((IM, IM, 3, b))

    Positions_compare = np.zeros((JM, IM, 3, b))

    Positions_OLD[:, :, 0, :] = array_Xprs
    Positions_OLD[:, :, 1, :] = array_Yprs
    Positions_OLD[:, :, 2, :] = array_Zprs

    Positions_compare[:] = Positions_OLD

    C_psi = 0
    RMS_array = []
    Power_array_iter = []
    Thrust_array_iter = []

    rms_change = 1
    Rotation_count = 0
    rotation = 0
    count = 0
    az_index = 0
    dbeta_dt = 0
    beta = 0
    Azimuthal_dT_array = np.zeros(N)
    Azimuthal_dOPi_array = np.zeros(N)
    Azimuthal_dOPo_array = np.zeros(N)

    last_time = time.time()

    circ_iter = np.tile(circulation, (b, 1))

    while rms_change > 0.01:
        count = count + 1
        rotation = rotation + dpsi * (180 / pi)
        V_IND = np.zeros((JM, 3, 3, blades))
        CirVM_array = np.zeros(IM)
        Blade_geometry = np.zeros((IM, 3, b))
        Blade_Velocities = np.zeros((IM, 3, b))
        CirVM_array[:] = circulation
        C_psi = C_psi + 1
        Radial_dT_array = np.zeros(IM)
        Radial_dOPi_array = np.zeros(IM)
        Radial_dOPo_array = np.zeros(IM)

        v_ind_obj = calc_VIND_obj(circ_iter, Positions_OLD, surface.cps)
        surface.set_v_ind(v_ind_obj)
        solver.solve(time_step, 0)

        V_IND_by_obj = np.zeros((JM, 3, 3, blades))
        for bl in range(1, b + 1):
            for ir in range(IM):
                Positions_NEW[0][ir][1][bl - 1] = radius[ir] * (
                    (cos(-1 * count * dpsi + ((bl - 1) / b) * 2 * pi) * cos(beta))
                    + sin(beta) * sin(0)
                )
                Positions_NEW[0][ir][0][bl - 1] = radius[ir] * (
                    sin(-1 * count * dpsi + ((bl - 1) / b) * 2 * pi) * cos(beta)
                )
                Positions_NEW[0][ir][2][bl - 1] = (
                    radius[ir]
                    * (
                        sin(beta) * cos(0)
                        - cos(-1 * count * dpsi + ((bl - 1) / b) * 2 * pi)
                        * cos(beta)
                        * sin(0)
                    )
                    + Positions_OLD[0][ir][2][bl - 1]
                )

                for j in range(JM):
                    V_IND_by_obj[j, :, 0, bl - 1] = solver.compute_total_vel(
                        Positions_OLD[j, ir, :, bl - 1]
                    )

                V_IND[:, :, 0, bl - 1] = (
                    Biot_Savarts_Law2(Positions_OLD, CirVM_array, bl - 1, ir)
                    + V_IND_by_obj[:, :, 0, bl-1]
                )

                Positions_NEW[1:, ir, :, bl - 1] = Positions_OLD[
                    0 : JM - 1, ir, :, bl - 1
                ] + (dpsi / omega) * (
                    V_INF[1:, :]
                    + (1 / 3)
                    * (
                        V_IND[1:, :, 0, bl - 1]
                        + V_IND[1:, :, 1, bl - 1]
                        + V_IND[1:, :, 2, bl - 1]
                    )
                )

                V_IND[:, :, 2, bl - 1] = V_IND[:, :, 1, bl - 1]
                V_IND[:, :, 1, bl - 1] = V_IND[:, :, 0, bl - 1]

                # for bl in range(1,b+1):
                #     for ir in range(IM):
                Blade_Velocities[ir, :, bl - 1] = Biot_Savarts_Perfromance(
                    Positions_NEW, CirVM_array, bl - 1, ir
                )
                psi = 1 * count * dpsi + ((bl - 1) / b) * 2 * pi
                theta_equation = theta[ir] + theta_1c * cos(psi) + theta_1s * sin(psi)
                Up = (
                    abs(Blade_Velocities[ir, 2, bl - 1])
                    + u_inf * sin(alpha)
                    + u_inf * sin(beta) * cos(psi)
                )

                # Up=   omega*R_tip*LAMBDA_I(radius[ir], psi, L_ig_calculated , mu )  + u_inf*sin(alpha) + u_inf*sin(beta)*cos(psi)
                Ut = omega * radius[ir] + u_inf * cos(alpha) * sin(psi)

                L_bar = (
                    0.5 * rho * chord[ir] * a * (theta_equation * Ut**2 - Ut * Up)
                )  # filling the  corresponding array
                CirVM_array[ir] = L_bar / (rho * Ut)
                # CirVM_array[IM-1]=0
                dDi = (
                    1.5
                    * 0.5
                    * rho
                    * chord[ir]
                    * a
                    * (theta_equation - atan(Up / Ut))
                    * Up
                    * Ut
                )  # Di term in induced torque equation
                dDo = (
                    0.5 * rho * Ut * Ut * chord[ir] * (Cdo)
                )  # Do term in  profile torque equation

                dPi = dDi * Ut  # filling the  corresponding array
                dPo = dDo * Ut
                Radial_dT_array[ir] = Radial_dT_array[ir] + L_bar
                Radial_dOPi_array[ir] = Radial_dOPi_array[ir] + dPi
                Radial_dOPo_array[ir] = Radial_dOPo_array[ir] + dPo

            circ_iter[bl - 1, :] = CirVM_array
        
        solver.finalize_iteration()

        Positions_OLD[:] = Positions_NEW
        dT_az = np.trapz(Radial_dT_array, radius)
        dPi_az = np.trapz(Radial_dOPi_array, radius)
        dPo_az = np.trapz(Radial_dOPo_array, radius)
        Azimuthal_dT_array[az_index] = dT_az * (1 / (2 * pi))
        Azimuthal_dOPi_array[az_index] = dPi_az * (1 / (2 * pi))
        Azimuthal_dOPo_array[az_index] = dPo_az * (1 / (2 * pi))
        circulation[:] = CirVM_array
        az_index = az_index + 1

        # print(az_index)

        if (C_psi % N) == 0:
            print("Time taken for the iteration: ", time.time() - last_time)
            last_time = time.time()
            # print("HI")
            Rotation_count += 1
            Change_array = (
                Positions_NEW[:, IM - 1, :, 0] - Positions_compare[:, IM - 1, :, 0]
            )
            dx2_dy2_dz2 = np.square(Change_array)
            dx2_dy2_dz2_sum = dx2_dy2_dz2[:, 0] + dx2_dy2_dz2[:, 1] + dx2_dy2_dz2[:, 2]
            # Eucladian_distance= np.sqrt(dx2_dy2_dz2[:,0]+ dx2_dy2_dz2[:,1]+dx2_dy2_dz2[:,2])
            shift_sum = np.sum(dx2_dy2_dz2_sum)
            err1 = sqrt((shift_sum) / JM)
            rms_change = err1 / R_tip
            RMS_array.append(rms_change)

            Thrust_new = np.trapz(Azimuthal_dT_array, psi_array)
            Power_new = abs(np.trapz(Azimuthal_dOPi_array, psi_array)) + np.trapz(
                Azimuthal_dOPo_array, psi_array
            )

            Azimuthal_dT_array = np.zeros(N)
            Azimuthal_dOPi_array = np.zeros(N)
            Azimuthal_dOPo_array = np.zeros(N)

            Power_array_iter.append(Power_new)
            Thrust_array_iter.append(Thrust_new)

            print(Rotation_count, rms_change, Thrust_new, Power_new * 0.001)
            Positions_compare[:] = Positions_NEW

            az_index = 0

        if Rotation_count == N_count:

            break

    # plt.plot(Power_array_iter)
    # plt.plot(Thrust_array_iter)

    return (
        Positions_NEW,
        RMS_array,
        CirVM_array,
        Thrust_new,
        Power_new,
        Blade_Velocities,
    )


# Blade_Velocities[i,:,bl-1]= Biot_Savarts_Perfromance(Positions_NEW, CirVM_array, bl-1, i)
# Up = abs(Blade_Velocities[i,2,bl-1])+ u_inf*sin(alpha) + u_inf*sin(beta)*cos(count*dpsi)  # perpendicular velocity

# %%
print(
    "Thrust(BEM_A2) N: {}  Power(BEM_A2) kW: {}".format(
        round(Thrust, 3), round(Power / 1000, 3)
    )
)
print(
    "Thrust(BEM_A3) N: {}  Power(BEM_A3) kW: {}".format(
        round(THRUST, 3), round(TOTAL_POWER / 1000, 3)
    )
)
current_time1 = time.time()
print(">>>>EX2B {}deg".format(round(dpsi_d, 2)))
print("   RMS_CHANGE         Thrust(N)       Power(kW)")
POS_EX2B, RMS_EX2B, Circ_updated, Thrust_updated, Power_updated, BV = EX2BMethod()
current_time2 = time.time()
Execution_time = round(current_time2 - current_time1, 5)
print("Execution time (EX2B Method ):", Execution_time, "sec")

#
plt.figure(10, dpi=150)
plt.plot(radius / R_tip, Circ_updated, "r-", label=" VM code")
plt.plot(radius / R_tip, circulation_old, "b.-", label=" BEM")
plt.title("Circulation Vs Radial location")
plt.xlabel("r/R")
plt.ylabel("Circulation")
plt.legend()

print(
    "Thrust(BEM): {} Power(BEM): {}".format(
        round(THRUST, 3), round(TOTAL_POWER / 1000, 3)
    )
)
print(
    "Thrust(VM): {} Power(VM): {} ".format(
        round(Thrust_updated, 3), round(Power_updated / 1000, 3)
    )
)
Difference = abs((Thrust_updated - Thrust) / Thrust_updated) * 100
print(Difference)
# %%
# plt.plot(RMS_PC2B[:])

# POSNr_explicit= np.divide(POS_explicit,R_tip)
# POSNr_EX2B= np.divide(POS_EX2B,R_tip)

# POSNr_explicit_E2B= np.divide(POS_explicit_E2B,R_tip)
POSNr_EX2B = np.divide(POS_EX2B, R_tip)
# POSNr_PC2B= np.divide(POS_PC2B,R_tip)

array_Xprsnr, array_Yprsnr, array_Zprsnr = (
    np.divide(array_Xprs, R_tip),
    np.divide(array_Yprs, R_tip),
    np.divide(array_Zprs, R_tip),
)

# %%
plt.figure(3, figsize=(5, 5), dpi=150)
# plt.figure(4,dpi=150)
ax = plt.axes(projection="3d")
# ax.set_title("Tip Vortex location")
ax.set_xlabel(r"$\frac{r}{R}$", fontsize=15)
ax.set_ylabel(r"$\frac{r}{R}$", fontsize=15)
ax.set_zlabel(r"$\frac{z}{R}$", fontsize=15)

# ax.set_ylim3d(-0.5*max(POS_explicit[:,0,:]),0.5*max(POS_explicit[:,0,:]))
if mu == 0:
    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-2, 2)

if mu > 0:
    ax.set_xlim3d(-2, 2)
    ax.set_zlim3d(-5, 1)
ax.view_init(0, 0)
for i in range(0, b):
    # plt.plot(array_Xprsnr[:, IM - 1, i], array_Yprsnr[:, IM - 1, i], array_Zprsnr[:, IM - 1, i], label='Prescibed', color='blue', marker='')

    plt.plot(
        POSNr_EX2B[:, IM - 1, 0, i],
        POSNr_EX2B[:, IM - 1, 1, i],
        POSNr_EX2B[:, IM - 1, 2, i],
        color="brown",
        label="EX2B",
        marker="",
        markersize=3,
        linewidth=1,
    )

    # plt.plot(POSNr_PC2B[:,IM-1,0,i],POSNr_PC2B[:,IM-1,1,i],POSNr_PC2B[:,IM-1,2,i],color='black',label='PC2B',marker='',markersize=3, linewidth=1)
plt.title(" Advance ratio {}   , CT {}".format(round(mu, 3), round(CT, 3)))
ax.legend()
plt.show()

# %%
plt.figure(4, dpi=150)
plt.plot(wake_age_array, array_Zprsnr[:, IM - 1, b - 1], "b-", label="Prescribed")
# plt.plot(POSNr_EX2B[:, IM - 1, 2, 0], 'r-.', label='EX2B')
# plt.plot(POSNr_PC2B[:,IM-1,2,0],'k-.',label='PC2B')
plt.plot(wake_age_array, POSNr_EX2B[:, IM - 1, 2, b - 1], "k", label="1")
# plt.plot(wake_age_array,POSNr_PC2B[:,IM-1,2,b-1],'r',label='PC2B')
plt.legend()

# %%
End_time = time.time()
print("Total Execution time:", round(End_time - Start_time, 5), "sec")

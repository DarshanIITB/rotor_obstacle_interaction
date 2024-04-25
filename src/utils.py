import numba
import numpy as np
import sympy as sp
from math import *
from constants import *

def set_values(matrix, i, n_panels, col: np.ndarray, val: np.ndarray):
    """
    Set values in a matrix similar to the PETSc MatSetValues function.

    Parameters:
    matrix (np.ndarray): The matrix to set values in.
    i (int): The row index to set the value.
    n_panels (int): The number of panels, which determines the number of columns.
    col (np.ndarray): The column indices where the values will be set.
    val (np.ndarray): The values to be set in the matrix.
    """

    # Check if the indices and values are of the same length
    if len(col) != len(val):
        raise ValueError("The length of 'col' and 'val' must be the same.")

    # Set the values in the matrix
    matrix[i, col] = val


L_ig = sp.Symbol("lambda_ig")  # declaring symbol lambda_ig
L_i = sp.Symbol("lambda_i")


@numba.njit
def Rotate(x2, y2, theta):
    X = x2 * np.cos(theta) - y2 * np.sin(theta)
    Y = x2 * np.sin(theta) + y2 * np.cos(theta)
    return X, Y


def Distance(x1, y1, z1, x2, y2, z2):
    dis = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
    return dis


alp = 1.2
K_visc = 0.000015
del_s = 1000


# %%
def LAMBDA_I(r_in, psi_in, L_ig_in, mu_in):
    LHS = (L_i) / (L_ig_in)  # LHS
    RHS = 1 + ((1.333 * mu_in) / (1.2 * (L_i + L_ig_in) + mu_in)) * (
        (r_in * cos(psi_in)) / R_tip
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
    return L_i_calculated


@numba.njit
def Biot_Savarts_Law2(Positions, circulation_in, b, IM_in):
    V_ind = np.zeros((JM, 3))

    for j in range(JM):
        xc = Positions[j][IM_in][0][b]
        yc = Positions[j][IM_in][1][b]
        zc = Positions[j][IM_in][2][b]
        for bl in range(1, blades + 1):
            for i in range(IM):
                if i == (IM - 1):
                    d_gamma = 0.8 * max(circulation_in)
                else:
                    d_gamma = circulation_in[i] - circulation_in[i + 1]
                for k in range(0, JM - 1):
                    if k != j and (k + 1) != j:
                        xa = Positions[k][i][0][bl - 1]
                        xb = Positions[k + 1][i][0][bl - 1]

                        ya = Positions[k][i][1][bl - 1]
                        yb = Positions[k + 1][i][1][bl - 1]

                        za = Positions[k][i][2][bl - 1]
                        zb = Positions[k + 1][i][2][bl - 1]
                        L = ((xa - xb) ** 2 + (ya - yb) ** 2 + (za - zb) ** 2) ** 0.5

                        cutoff_radius = (0.25 * L) ** 2

                        V_induced = np.zeros(3)
                        L = ((xa - xb) ** 2 + (ya - yb) ** 2 + (za - zb) ** 2) ** 0.5

                        cutoff = sqrt(
                            (0.25 * L) ** 4
                            + 4 * alp * del_s * K_visc * ((k + 1) * dpsi / omega)
                        )
                        r1 = (
                            (xc - xa) ** 2 + (yc - ya) ** 2 + (zc - za) ** 2
                        ) ** 0.5 + cutoff
                        r2 = (
                            (xc - xb) ** 2 + (yc - yb) ** 2 + (zc - zb) ** 2
                        ) ** 0.5 + cutoff

                        denom = 2 * pi * r1 * r2 * ((r1 + r2) ** 2 - L**2)

                        TERM1 = (
                            d_gamma
                            * (r1 + r2)
                            / (2 * pi * (r1) * (r2) * ((r1 + r2) ** 2 - L**2))
                        )
                        u_bs = TERM1 * ((zc - zb) * (yb - ya) - (yc - yb) * (zb - za))
                        v_bs = TERM1 * ((xc - xb) * (zb - za) - (zc - zb) * (xb - xa))
                        w_bs = TERM1 * ((yc - yb) * (xb - xa) - (xc - xb) * (yb - ya))
                        V_induced[0] = u_bs
                        V_induced[1] = v_bs
                        V_induced[2] = w_bs

                        V_add = V_induced  #

                    else:
                        V_add = np.zeros(3)
                    V_ind[j, :] = V_ind[j, :] + V_add
    return V_ind


# %%
@numba.njit
def Biot_Savarts_Perfromance(
    Positions, circulation_in, b, IM_in
):  # recheck this part for
    V_ind = np.zeros(3)
    xc = Positions[0][IM_in][0][b]
    yc = Positions[0][IM_in][1][b]
    zc = Positions[0][IM_in][2][b]
    for bl in range(1, blades + 1):
        for i in range(IM):
            if i == (IM - 1):
                d_gamma = 0.8 * max(circulation_in)
            else:
                d_gamma = circulation_in[i] - circulation_in[i + 1]
            for k in range(0, JM - 1):
                xa = Positions[k][i][0][bl - 1]
                xb = Positions[k + 1][i][0][bl - 1]

                ya = Positions[k][i][1][bl - 1]
                yb = Positions[k + 1][i][1][bl - 1]

                za = Positions[k][i][2][bl - 1]
                zb = Positions[k + 1][i][2][bl - 1]
                L = ((xa - xb) ** 2 + (ya - yb) ** 2 + (za - zb) ** 2) ** 0.5

                cutoff_radius = (0.25 * L) ** 2

                V_induced = np.zeros(3)
                L = ((xa - xb) ** 2 + (ya - yb) ** 2 + (za - zb) ** 2) ** 0.5

                cutoff = sqrt(
                    (0.25 * L) ** 4
                    + 4 * alp * del_s * K_visc * ((k + 1) * dpsi / omega)
                )
                r1 = ((xc - xa) ** 2 + (yc - ya) ** 2 + (zc - za) ** 2) ** 0.5 + cutoff
                r2 = ((xc - xb) ** 2 + (yc - yb) ** 2 + (zc - zb) ** 2) ** 0.5 + cutoff

                denom = 2 * pi * r1 * r2 * ((r1 + r2) ** 2 - L**2)

                TERM1 = (
                    d_gamma
                    * (r1 + r2)
                    / (2 * pi * (r1) * (r2) * ((r1 + r2) ** 2 - L**2))
                )
                u_bs = TERM1 * ((zc - zb) * (yb - ya) - (yc - yb) * (zb - za))
                v_bs = TERM1 * ((xc - xb) * (zb - za) - (zc - zb) * (xb - xa))
                w_bs = TERM1 * ((yc - yb) * (xb - xa) - (xc - xb) * (yb - ya))
                V_induced[0] = u_bs
                V_induced[1] = v_bs
                V_induced[2] = w_bs

                V_add = V_induced  #
                V_ind[:] = V_ind[:] + V_add
    return V_ind

def calc_VIND_obj(circulation: np.ndarray, Positions: np.ndarray, panel_cps: np.ndarray):
    """
    Calculate induced velocity due to the rotor wake at the object surface
    Args:
        circulation (np.ndarray): circulations on the rotor wake
        Positions (np.ndarray): positions of the rotor wake
        panel_cps (np.ndarray): positions of the elements on the object surface
    Returns:
        np.ndarray: Induced velocity at the object surface due to the rotor wake
    """
    V_ind = np.zeros((len(panel_cps), 3))
    for j in range(len(panel_cps)):
        xc = panel_cps[j][0]
        yc = panel_cps[j][1]
        zc = panel_cps[j][2]
        for bl in range(1, blades + 1):
            for i in range(IM):
                if i == (IM - 1):
                    d_gamma = 0.8 * max(circulation[bl-1])
                else:
                    d_gamma = circulation[bl-1][i] - circulation[bl-1][i + 1]
                for k in range(0, JM-1):
                    xa = Positions[k][i][0][bl - 1]
                    xb = Positions[k + 1][i][0][bl - 1]

                    ya = Positions[k][i][1][bl - 1]
                    yb = Positions[k + 1][i][1][bl - 1]

                    za = Positions[k][i][2][bl - 1]
                    zb = Positions[k + 1][i][2][bl - 1]
                    L = ((xa - xb) ** 2 + (ya - yb) ** 2 + (za - zb) ** 2) ** 0.5

                    cutoff_radius = (0.25 * L) ** 2

                    V_induced = np.zeros(3)
                    L = ((xa - xb) ** 2 + (ya - yb) ** 2 + (za - zb) ** 2) ** 0.5

                    cutoff = sqrt(
                        (0.25 * L) ** 4
                        + 4 * alp * del_s * K_visc * ((k + 1) * dpsi / omega)
                    )
                    r1 = (
                        (xc - xa) ** 2 + (yc - ya) ** 2 + (zc - za) ** 2
                    ) ** 0.5 + cutoff
                    r2 = (
                        (xc - xb) ** 2 + (yc - yb) ** 2 + (zc - zb) ** 2
                    ) ** 0.5 + cutoff

                    denom = 2 * pi * r1 * r2 * ((r1 + r2) ** 2 - L**2)

                    TERM1 = (
                        d_gamma
                        * (r1 + r2)
                        / (2 * pi * (r1) * (r2) * ((r1 + r2) ** 2 - L**2))
                    )

                    u_bs = TERM1 * ((zc - zb) * (yb - ya) - (yc - yb) * (zb - za))
                    v_bs = TERM1 * ((xc - xb) * (zb - za) - (zc - zb) * (xb - xa))
                    w_bs = TERM1 * ((yc - yb) * (xb - xa) - (xc - xb) * (yb - ya))

                    V_induced[0] = u_bs
                    V_induced[1] = v_bs
                    V_induced[2] = w_bs

                    V_add = V_induced  #
                    V_ind[j, :] = V_ind[j, :] + V_add
    return V_ind
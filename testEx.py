import numpy as np
import sympy as smp


def RotX(theta):
    return smp.Matrix([[1, 0, 0], [0, smp.cos(theta), -
                                   smp.sin(theta)], [0, smp.sin(theta), smp.cos(theta)]])


def RotY(theta):
    return smp.Matrix([[smp.cos(theta), 0, smp.sin(theta)], [
        0, 1, 0], [-smp.sin(theta, 0, smp.cos(theta))]])


def RotZ(theta):
    return smp.Matrix([[smp.cos(theta), -smp.sin(theta), 0],
                       [smp.sin(theta), smp.cos(theta), 0], [0, 0, 1]])


def HomogeneousMatrix(R, p):
    return smp.BlockMatrix([[R, p]])


L1x, L1y, L1z = smp.symbols("L1x L1y L1z")
L2x, L2y, L2z = smp.symbols("L2x L2y L2z")
L3x, L3y, L3z = smp.symbols("L3x L3y L3z")
Lhx, Lhy, Lhz = smp.symbols("Lhx Lhy Lhz")

theta1, theta2, theta3 = smp.symbols("theta1 theta2 theta3")

R01 = RotZ(-theta1)
p01 = smp.Matrix([[L1x], [L1y], [L1z]])

T01 = HomogeneousMatrix(R01, p01)

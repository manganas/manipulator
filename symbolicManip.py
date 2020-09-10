import numpy as np
import sympy as smp


def RotX(theta):
    return smp.Matrix([[1, 0, 0], [0, smp.cos(theta), -
                                   smp.sin(theta)], [0, smp.sin(theta), smp.cos(theta)]])


def RotY(theta):
    return smp.Matrix([[smp.cos(theta), 0, smp.sin(theta)], [
        0, 1, 0], [-smp.sin(theta), 0, smp.cos(theta)]])


def RotZ(theta):
    return smp.Matrix([[smp.cos(theta), -smp.sin(theta), 0],
                       [smp.sin(theta), smp.cos(theta), 0], [0, 0, 1]])


def HomogeneousMatrix(R, p):
    p1 = smp.Matrix(p).reshape(3, 1)
    return smp.Matrix(smp.BlockMatrix([[R, p1], [smp.zeros(1, 3), smp.Matrix([1])]]))


th = smp.symbols("theta")

R = RotZ(th)
p = smp.Matrix([1, 2, 3])

import numpy as np
import sympy as smp

import manip
import reachAlphaSymbolic as manipSym

d1x, d1z, d2x, d2z, d3x, d3z = smp.symbols(
    "d_1x d_1z d_2x d_2z d_3x d_3z")  # all positive
th1, th2, th3 = smp.symbols("theta1 theta2 theta3")

T01 = manipSym.HomogeneousRotOnly(manipSym.RotZ(th1))

T12 = manipSym.HomogeneousTransOnly(
    [d1x, 0, -d1z])*manipSym.HomogeneousRotOnly(manipSym.RotY(th2-smp.pi/2.0))

T23 = manipSym.HomogeneousTransOnly(
    [d2x, 0, -d2z])*manipSym.HomogeneousRotOnly(manipSym.RotY(-(th3+smp.pi)))

T3h = manipSym.HomogeneousTransOnly([d3x, 0, d3z])

T0h = smp.simplify(T01*T12*T23*T3h)
T0h.subs([(th1, 0), (th2, smp.pi/2), (th3, smp.pi)])

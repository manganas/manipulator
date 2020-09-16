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


def Displace(d_row):
    d1, d2, d3 = d_row
    I = smp.Matrix.eye(3)
    Z = smp.zeros(1, 3)
    p = smp.Matrix([[d1], [d2], [d3]])
    blockMatrix = smp.BlockMatrix([[I, p], [Z, smp.Matrix([1])]])
    return smp.Matrix(blockMatrix)


def HomogeneousRotOnly(R):
    blockMatrix = smp.BlockMatrix(
        [[R, smp.zeros(3, 1)], [smp.zeros(1, 3), smp.Matrix([1])]])
    return smp.Matrix(blockMatrix)


def HomogeneousTransOnly(p):
    p1 = smp.Matrix(p).reshape(3, 1)
    R = smp.Matrix.eye(3)
    blockMatrix = smp.BlockMatrix(
        [[R, p1], [smp.zeros(1, 3), smp.Matrix([1])]])
    return smp.Matrix(blockMatrix)


def HomogeneousMatrix(R, p):
    p1 = smp.Matrix(p).reshape(3, 1)
    return smp.Matrix(smp.BlockMatrix([[R, p1], [smp.zeros(1, 3), smp.Matrix([1])]]))


def HomogeneousInverse(T):
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    R_inv = smp.Matrix.transpose(R)
    p_inv = -R_inv*p
    blockMatrix = smp.BlockMatrix(
        [[R_inv, p_inv], [smp.zeros(1, 3),  smp.Matrix([1])]])
    return smp.Matrix(blockMatrix)


def SkewSymmetric(p_row):
    p1, p2, p3 = p_row
    return smp.Matrix([[0, -p3, p2], [p3, 0, -p1], [-p2, p1, 0]])


def SkewToVec(S):
    return smp.Matrix([[S[2, 1]], [-S[2, 0]], [S[1, 0]]])


def MatrixEquality(A, B, tolerance):
    D = A-B
    for element in D:
        if smp.Abs(element) > tolerance:
            return False
    return True


def MatrixExponentialSO3(w_row, theta):
    W = SkewSymmetric(w_row)
    return smp.Matrix.eye(3) + smp.sin(theta)*W + (1-smp.cos(theta))*W*W


def se3ToTwist(se3):
    wSkew = se3[0:3, 0:3]
    w = SkewToVec(wSkew)
    v = se3[0:3, 3].reshape(3, 1)
    blockMatrix = smp.BlockMatrix([[w], [v]])
    return smp.Matrix(blockMatrix)


def ScrewTose3(S):
    S1 = smp.Matrix(S).reshape(6, 1)
    w = S1[0:3, 0].reshape(1, 3)
    v = S1[3:, 0].reshape(3, 1)
    wSkew = SkewSymmetric(w)
    blockMatrix = smp.BlockMatrix(
        [[wSkew, v], [smp.zeros(1, 3), smp.Matrix([0])]])
    return smp.Matrix(blockMatrix)


def Adjoint(T):
    R = T[0:3, 0:3]
    p = T[0:3, 3].reshape(3, 1)
    pSkew = SkewSymmetric(p)
    blockMatrix = smp.BlockMatrix([R, smp.zeros(3, 3), [pSkew*R, R]])
    return smp.Matrix(blockMatrix)


# Reach Alpha parameters
L1x, L1y, L1z = smp.symbols("L_1x L_1y L_1z")
L2x, L2y, L2z = smp.symbols("L_2x L_2y L_2z")
L3x, L3y, L3z = smp.symbols("L_3x L_3y L_3z")
Lhx, Lhy, Lhz = smp.symbols("L_hx L_hy L_hz")

th1, th2, th3 = smp.symbols("theta1 theta2 theta3")

R01 = RotZ(-th1)
p01 = smp.Matrix([L1x, 0, L1z])
T01 = HomogeneousMatrix(R01, p01)

R12 = RotY(th2)
p12 = smp.Matrix([L2x, 0, L2z])
T12 = HomogeneousMatrix(R12, p12)

R23 = RotY(th3)
p23 = smp.Matrix([L3x, 0, L3z])
T23 = HomogeneousMatrix(R23, p23)

R3h = RotY(smp.pi/2.0)
p3h = smp.Matrix([Lhx, 0, Lhz])
T3h = HomogeneousMatrix(R3h, p3h)

T0h = smp.simplify(T01@T12@T23@T3h)


# def forwardKinematics(theta_array, p_array):
#     if len(theta_array) == 3:
#         theta1, theta2, theta3 = theta_array
#     else:
#         print("Need an array of exactly 3 angles")
#         return None
#     if len(p_array) != 3:
#         print("Mistake in frame")
#         return None
#     return np.array([[-np.sin(theta2 + theta3)*np.cos(theta1), np.sin(theta1),
#                       np.cos(theta1)*np.cos(theta2 + theta3),
#                       L_1x + L_2x*np.cos(theta1) + L_3x*np.cos(theta1)*np.cos(theta2) + L_3z*np.sin(theta2)*np.cos(theta1) + L_hx*np.cos(theta1)*np.cos(theta2 + theta3) + L_hz*np.sin(theta2 + theta3)*np.cos(theta1)],
#                      [np.sin(theta1)*np.sin(theta2 + theta3), np.cos(theta1),
#                       -np.sin(theta1)*np.cos(theta2 + theta3),
#                       -(L_2x + L_3x*np.cos(theta2) + L_3z*np.sin(theta2) + L_hx*np.cos(theta2 + theta3) + L_hz*np.sin(theta2 + theta3))*np.sin(theta1)],
#                      [-np.cos(theta2 + theta3), 0, -np.sin(theta2 + theta3),
#                       L_1z + L_2z - L_3x*np.sin(theta2) + L_3z*np.cos(theta2) - L_hx*np.sin(theta2 + theta3) + L_hz*np.cos(theta2 + theta3)],
#                      [0, 0, 0, 1]], dtype=object)

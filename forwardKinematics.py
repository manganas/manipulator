import numpy as np

import manip


def cos(theta):
    return np.cos(theta)


def sin(theta):
    return np.sin(theta)


def tan(theta):
    return np.tan(theta)


def atan2(y, x):
    return np.arctan2(y, x)


l__1x = 0
l__1y = 0
l__1z = 0.05

l__2x = 0.25
l__2y = 0
l__2z = 0

l__3x = 0.2
l__3y = 0
l__3z = 0

l__hx = 0.08
l__hy = 0
l__hz = 0


def forwardKinematicsPosition(thetas_array):
    thetas = np.array(thetas_array).reshape(3, 1)
    theta1 = thetas[0, 0]
    theta2 = thetas[1, 0]
    theta3 = thetas[2, 0]
    xp = ((cos(theta3)*l__hx + sin(theta3)*l__hz + l__3x)*cos(theta2) + (cos(theta3)*l__hz - sin(theta3)
                                                                         * l__hx + l__3z)*sin(theta2) + l__2x)*cos(theta1) + (-l__2y - l__3y - l__hy)*sin(theta1) + l__1x
    yp = ((cos(theta3)*l__hx + sin(theta3)*l__hz + l__3x)*cos(theta2) + (cos(theta3)*l__hz - sin(theta3)
                                                                         * l__hx + l__3z)*sin(theta2) + l__2x)*sin(theta1) + (l__2y + l__3y + l__hy)*cos(theta1) + l__1y
    zp = (cos(theta3)*l__hz - sin(theta3)*l__hx + l__3z)*cos(theta2) + \
        (-cos(theta3)*l__hx - sin(theta3) *
         l__hz - l__3x)*sin(theta2) + l__1z + l__2z
    return np.array([xp, yp, zp]).reshape(3, 1)


def JacobianForwardKinematicsPosition(thetas_array):
    thetas = np.array(thetas_array).reshape(3, 1)
    theta1 = thetas[0, 0]
    theta2 = thetas[1, 0]
    theta3 = thetas[2, 0]
    J11 = (cos(theta3)*l__hz - sin(theta3)*l__hx + l__3z)*cos(theta2) + \
        (-cos(theta3)*l__hx - sin(theta3) *
         l__hz - l__3x)*sin(theta2) + l__1z + l__2z
    J12 = (-(cos(theta3)*l__hx + sin(theta3)*l__hz + l__3x)*sin(theta2) +
           (cos(theta3)*l__hz - sin(theta3)*l__hx + l__3z)*cos(theta2))*cos(theta1)
    J13 = ((cos(theta3)*l__hz - sin(theta3)*l__hx)*cos(theta2) +
           (-sin(theta3)*l__hz - cos(theta3)*l__hx)*sin(theta2))*cos(theta1)
    J21 = ((cos(theta3)*l__hx + sin(theta3)*l__hz + l__3x)*cos(theta2) + (cos(theta3)*l__hz -
                                                                          sin(theta3)*l__hx + l__3z)*sin(theta2) + l__2x)*cos(theta1) - (l__2y + l__3y + l__hy)*sin(theta1)
    J22 = (-(cos(theta3)*l__hx + sin(theta3)*l__hz + l__3x)*sin(theta2) +
           (cos(theta3)*l__hz - sin(theta3)*l__hx + l__3z)*cos(theta2))*sin(theta1)
    J23 = ((cos(theta3)*l__hz - sin(theta3)*l__hx)*cos(theta2) +
           (-sin(theta3)*l__hz - cos(theta3)*l__hx)*sin(theta2))*sin(theta1)
    J31 = 0
    J32 = -(cos(theta3)*l__hz - sin(theta3)*l__hx + l__3z)*sin(theta2) + \
        (-cos(theta3)*l__hx - sin(theta3)*l__hz - l__3x)*cos(theta2)
    J33 = (-sin(theta3)*l__hz - cos(theta3)*l__hx)*cos(theta2) + \
        (sin(theta3)*l__hx - cos(theta3)*l__hz)*sin(theta2)
    J = np.array([[J11, J12, J13], [J21, J22, J23], [J31, J32, J33]])
    return J

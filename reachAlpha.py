import numpy as np

import manip

d1x = 0.02
d1z = 0.03
l2 = 0.15
l3 = 0.12


def InverseKinematicsUp(p_ref_array):
    x, y, z = p_ref_array
    theta1 = np.arctan2(y, x)
    r = np.sqrt(x**2+y**2)
    r1 = r-d1x

    r3 = np.sqrt(r1**2 + (z-d1z)**2)
    phi1 = np.arctan2(z-d1z, r1)
    cphi2 = (l2**2 + r3**2-l3**2)/(2*l2*r3)
    sphi2 = np.sqrt(1-cphi2**2)
    phi2 = np.arctan2(sphi2, cphi2)
    theta2 = phi1+phi2

    cphi3 = (l2**2+l3**2-r3**2)/(2*l2*l3)
    sphi3 = np.sqrt(1-cphi3**2)
    phi3 = np.arctan2(sphi3, cphi3)
    theta3 = -(np.pi-phi3)
    return [theta1, theta2, theta3]


def InverseKinematicsDown(p_ref_array):
    x, y, z = p_ref_array
    theta1 = np.arctan2(y, x)
    r = np.sqrt(x**2+y**2)
    r1 = r-d1x

    r3 = np.sqrt(r1**2 + (z-d1z)**2)
    phi1 = np.arctan2(z-d1z, r1)
    cphi2 = (l2**2 + r3**2-l3**2)/(2*l2*r3)
    sphi2 = np.sqrt(1-cphi2**2)
    phi2 = np.arctan2(sphi2, cphi2)
    theta2 = (phi1-phi2)

    cphi3 = (l2**2+l3**2-r3**2)/(2*l2*l3)
    sphi3 = np.sqrt(1-cphi3**2)
    phi3 = np.arctan2(sphi3, cphi3)
    theta3 = (np.pi-phi3)
    return [theta1, theta2, theta3]


w1 = np.array([0, 0, 1])
p1 = np.array([0, 0, 0])

w2 = np.array([0, -1, 0])
p2 = np.array([d1x, 0, d1z])

w3 = np.array([0, -1, 0])
p3 = np.array([d1x+l2, 0, d1z])

R0 = np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
p0 = np.array([d1x+l2+l3, 0, d1z])
M0 = np.block([[R0, p0.reshape(3, 1)], [np.zeros((1, 3)), 1.0]])

theta1 = 10
theta2 = 20
theta3 = -30
thetas = np.array([theta1, theta2, theta3])*np.pi/180.0

w = [w1, w2, w3]
p = [p1, p2, p3]


fk = manip.ForwardKinematicsSpace(w, p, thetas, M0)
pd = fk[0:3, 3]

inUp = InverseKinematicsUp(pd)
inDown = InverseKinematicsDown(pd)

fkUp = manip.ForwardKinematicsSpace(w, p, inUp, M0)
fkDown = manip.ForwardKinematicsSpace(w, p, inDown, M0)

phUp = fkUp[0:3, 3]
phDown = fkDown[0:3, 3]

print(f"\nAngles [deg]:\n {[theta1, theta2, theta3]}\n")
print(f"\nForward kinematics:\n {np.round(fk,3)}\n")

print(
    f"\nAngles elbow Up [deg]:\n {np.round(np.array(inUp).reshape(1,3)*180.0/np.pi,3)}\n")
print(f"\nForward kinematics elbow Up:\n {np.round(fkUp,3)}\n")

print(
    f"\nAngles elbow Down [deg]:\n {np.round(np.array(inDown).reshape(1,3)*180.0/np.pi,3)}\n")
print(f"\nForward kinematics elbow Down:\n {np.round(fkDown,3)}\n")

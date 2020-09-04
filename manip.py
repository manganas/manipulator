import numpy as np


def RotX(theta):
    return np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])


def RotY(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])


def RotZ(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])


def Homog(R, p):
    p1 = np.array(p).reshape(3, 1)
    R1 = np.array(R)
    return np.block([[R1, p1], [np.zeros((1, 3)), 1]])


def HomogInv(T):
    R = T[0:3, 0:3]
    p = T[0:3, 3].reshape(3, 1)
    RTransp = np.transpose(R)
    return np.block([[RTransp, np.dot(-RTransp, p)], [np.zeros((1, 3)), 1]])


def skewSymmetric(p):
    p1 = np.array(p).reshape(3, 1)
    return np.array([[0, -p1[2, 0], p1[1, 0]], [p1[2, 0], 0, -p1[0, 0]], [-p1[1, 0], p1[0, 0], 0]])


def skewToVec(S):
    S1 = np.array(S)
    return np.array([S1[2, 1], -S1[2, 0], S1[1, 0]]).reshape(3, 1)


def matrixExpRot(w, theta):
    w1 = np.array(w).reshape(3, 1)
    W = skewSymmetric(w1)
    return np.eye(3) + np.sin(theta)*W + (1-np.cos(theta))*np.dot(W, W)


def matrixLogRot(R):
    # return an array of w:np.array and theta
    R1 = np.array(R)
    r11 = R1[0, 0]
    r12 = R1[0, 1]
    r13 = R1[0, 2]
    r21 = R1[1, 0]
    r23 = R1[1, 2]
    r22 = R1[1, 1]
    r31 = R1[2, 0]
    r32 = R1[2, 1]
    r33 = R1[2, 2]
    if matrixEquality(R1, np.eye(3), 000.1):
        theta = 0
        w = None
        return [w, theta]
    elif np.trace(R) == -1:
        theta = np.pi
        if r33 != -1:
            w = 1/np.sqrt(2*(1+r33))*np.array([r13, r23, 1+r33]).reshape(3, 1)
        elif r22 != -1:
            w = 1/np.sqrt(2*(1+r22))*np.array([r12, 1+r22, r32]).reshape(3, 1)
        else:
            w = 1/np.sqrt(2*(1+r11))*np.array([1+r11, r21, r31]).reshape(3, 1)
        return [w, theta]
    else:
        theta = np.arccos(0.5*(np.trace(R)-1))
        wSkew = 1/(2*np.sin(theta))*(R-np.transpose(R))
        w = skewToVec(wSkew).reshape(3, 1)
        return [w, theta]


def se3ToTwist(T):
    T1 = np.array(T)
    wSkew = T1[0:3, 0:3]
    w = skewToVec(wSkew).reshape(3, 1)
    v = T1[0:3, 3].reshape(3, 1)
    return np.block([[w], [v]])


def screwTose3(S):
    S1 = np.array(S).reshape(6, 1)
    w = S1[0:3, 0].reshape(3, 1)
    v = S1[3:, 0].reshape(3, 1)
    wSkew = skewSymmetric(w)
    return np.block([[wSkew, v], [np.zeros((1, 4))]])


def Adjoint(T):
    T1 = np.array(T)
    R = T1[0:3, 0:3]
    p = T1[0:3, 3].reshape(3, 1)
    pSkew = skewSymmetric(p)
    return np.block([[R, np.zeros((3, 3))], [np.dot(pSkew, R), R]])


def twoNorm(v):
    v1 = np.array(v).reshape(len(v), 1)
    sum = 0
    for element in v1:
        sum += element*element
    return np.sqrt(sum)[0]


def homogExp(S, theta):
    # S is a screw axis, not in se3 but in vector form
    S1 = np.array(S).reshape(6, 1)
    w = S1[0:3, 0].reshape(3, 1)
    v = S1[3:, 0].reshape(3, 1)
    if np.abs(twoNorm(w) - 1) < 0.001:
        wSkew = skewSymmetric(w)
        G = np.eye(3)*theta + (1 - np.cos(theta))*wSkew + \
            (theta - np.sin(theta))*np.dot(wSkew, wSkew)
        return np.block([[matrixExpRot(w, theta), np.dot(G, v)], [np.zeros((1, 3)), 1]])
    elif np.abs(w.all()) < 0.001 and np.abs(twoNorm(v) - 1) < 0.001:
        return np.block([[np.eye(3), v*theta], [np.zeros((1, 3)), 1]])
    else:
        return None


def matrixEquality(A, B, tol):
    A1 = np.array(A)
    B1 = np.array(B)
    if np.size(A1, 1) == np.size(B1, 1) and np.size(A1, 0) == np.size(B1, 0):
        C = np.abs(A-B)
        if C.any() > np.abs(tol):
            return False
        return True
    else:
        return None


def homogLog(T):
    T1 = np.array(T)
    R1 = T1[0:3, 0:3]
    p1 = T1[0:3, 3].reshape(3, 1)
    if matrixEquality(R1, np.eye(3), 000.1):
        w = np.array([0, 0, 0]).reshape(3, 1)
        v = p1*1/twoNorm(p1)
        v = v.reshape(3, 1)
        theta = twoNorm(p1)
        S = np.block([[w], [v]])
        return [S, theta]
    else:
        w, theta = matrixLogRot(R1)
        wSkew = skewSymmetric(w)
        invG = np.eye(3)*1/theta - 0.5*wSkew + (1/theta - 0.5 *
                                                (1/np.tan(theta/2))) * np.dot(wSkew, wSkew)
        v = np.dot(invG, p1)
        S = np.block([[w], [v]])
        return [S, theta]


def norm2(p):
    p1 = np.array(p)
    sum = 0
    for element in p1:
        sum += element**2
    return np.sqrt(sum)


def createS(w, p):  # include prismatic
    w1 = np.array(w).reshape(3, 1)
    p1 = np.array(p).reshape(3, 1)

    if norm2(w1) == 0:
        if norm2(p1) != 1:
            print("Something is wrong with the prismatic screw axis")
            return None
        return np.block([[w1], [p1]])
    wSkew = skewSymmetric(w1)
    v = -np.dot(wSkew, p1)
    return np.block([[w1], [v]])


def ScrewAxesSpace(w_array, p_array):
    w_length = len(w_array)
    p_length = len(p_array)
    if (w_length != p_length):
        print("w and p arrays do not have the same number of elements")
        return None

    S = np.zeros((6, w_length))
    for i in range(w_length):
        S[:, i] = createS(w_array[i], p_array[i]).reshape(1, 6)
    return S


def ScrewAxesBody(w_array, p_array, Minit):
    M0 = np.array(Minit)
    if (np.size(M0, 0) != 4 and np.size(M0, 1) != 4):
        print("Wrong home configuration homogeneous matrix")
        return None
    Adj = Adjoint(np.linalg.inv(M0))
    w_length = len(w_array)
    S = ScrewAxesSpace(w_array, p_array)
    B = np.zeros((6, w_length))
    for i in range(w_length):
        B[:, i] = np.dot(Adj, S[:, i].reshape(6, 1))
    return B


# Forward kinematics

def ForwardKinematicsSpace(w_array, p_array, theta_array, M0):
    S = ScrewAxesSpace(w_array, p_array)
    product = np.eye(4)
    w_length = len(w_array)
    for i in range(w_length):
        product = np.dot(product, homogExp(S[:, i], theta_array[i]))
    return np.dot(product, M0)


def ForwardKinematicsSpace_(S, theta_array, M0):
    product = np.eye(4)
    S_columns = np.size(S, 1)
    for i in range(S_columns):
        product = np.dot(product, homogExp(S[:, i], theta_array[i]))
    return np.dot(product, M0)


def ForwardKinematicsBody(w_array, p_array, theta_array, M0):
    B = ScrewAxesBody(w_array, p_array, M0)
    product = np.eye(4)
    w_length = len(w_array)
    for i in range(w_length):
        product = np.dot(product, homogExp(B[:, i], theta_array[i]))
    return np.dot(M0, product)


def ForwardKinematicsBody_(B, theta_array, M0):
    product = np.eye(4)
    B_columns = np.size(B, 1)
    for i in range(B_columns):
        product = np.dot(product, homogExp(B[:, i], theta_array[i]))
    return np.dot(M0, product)

# Velocity kinematics


def JacobianSpace(S, theta_array):
    S_columns = np.size(S, 1)
    theta_length = len(theta_array)
    if (S_columns != theta_length):
        print("Number of screw axes columns not equal to joint angles array elements.")
        return None
    J = np.zeros((6, S_columns))
    product = np.eye(4)
    for i in range(S_columns):
        if i == 0:
            J[:, i] = S[:, i].reshape(1, 6)
        else:
            product = np.dot(product, homogExp(S[:, i-1], theta_array[i-1]))
            Adj = Adjoint(product)
            J[:, i] = np.dot(Adj, S[:, i]).reshape(1, 6)
    return J


def JacobianBody(Tsb, Js):
    Tbs = HomogInv(Tsb)
    Adj = Adjoint(Tbs)
    return np.dot(Adj, Js)


# Numerical Inverse Kinematics
def Clamp(in_val, val_max, val_min):
    if in_val > val_max:
        return val_max
    if in_val < val_min:
        return val_min
    return in_val


def FixAngle(in_angle):
    '''
        Values in radians
    '''
    out_angle = in_angle
    while out_angle > 2*np.pi:
        out_angle -= 2*np.pi
    while out_angle < -2*np.pi:
        out_angle += 2*np.pi
    if out_angle < 0:
        out_angle += 2*np.pi
    return out_angle


def Geometric3dofArticulated(xRef, yRef, zRef):
    # define fixed geometric lengths
    l1 = 0.4
    l2 = 0.6
    d1 = 0.4

    # define limits for each axis
    theta1Max = 350*np.pi/180.0
    theta1Min = 0

    theta2Max = 200*np.pi/180.0
    theta2Min = 0

    theta3Max = 150*np.pi/180.0
    theta3Min = 0

    theta1 = np.arctan2(yRef, xRef)
    r3 = np.sqrt(xRef**2 + yRef**2)
    r2 = zRef - d1  # define d1, it is a fixed quantity
    r1 = np.sqrt(r3**2 - r2**2)
    phi3 = np.arccos((l1**2 + l2**2-r3**2)/(2*l1*l2))
    theta3 = np.pi-phi3
    phi2 = np.arctan2(r2, r1)
    phi1 = np.arccos((r3**2 + l1**2 - l2**2)/(2*r3*l1))
    theta2 = phi2 - phi1

    theta1 = FixAngle(theta1)
    theta2 = FixAngle(theta2)
    theta3 = FixAngle(theta3)

    theta1 = Clamp(theta1, theta1Max, theta1Min)
    theta2 = Clamp(theta2, theta2Max, theta2Min)
    theta3 = Clamp(theta3, theta3Max, theta3Min)
    return [theta1, theta2, theta3]


def my_functions(thetas):
    theta = np.array(thetas).reshape(len(thetas), 1)
    theta1 = theta[0, 0]
    theta2 = theta[1, 0]
    return np.array([[5*theta1**2 - np.sin(theta2)], [np.cos(theta1)-theta2**-1]])


def my_jacobian(thetas):
    theta = np.array(thetas).reshape(len(thetas), 1)
    theta1 = theta[0, 0]
    theta2 = theta[1, 0]
    return np.array([[10*theta1 - np.cos(theta2)], [-np.sin(theta1) + theta2**-2]])


def NRRootsScalar(function, Jacobian, xd, theta0, tol):
    i = 0
    theta_init = theta0
    e = xd - function(theta_init)
    while np.abs(e) > tol:
        theta_init += (Jacobian(theta_init))**-1 * (e)
        e = xd - function(theta_init)
        i += 1
    return [theta_init, i]


def NRRootsVector(functions, Jacobian, xd_array, theta0_array, tol):
    i = 0
    theta_init = np.array(theta0_array).reshape(len(theta0_array), 1)
    xd = np.array(xd_array).reshape(len(xd_array), 1)
    e = xd - theta_init
    while twoNorm(e) > tol:
        Jpinv = np.linalg.pinv(Jacobian(theta_init))
        theta_init += np.dot(Jpinv, e)
        e = xd - functions(theta_init)
        i += 1
    return [theta_init, i]


### Tests ###
Lx = 0.30
Lz = 0.65
Lh = 0.20

w1 = np.array([0, 0, 1])
p1 = np.array([0, 0, 0])

w2 = np.array([1, 0, 0])
p2 = np.array([0, 0, 0])

w3 = np.array([0, 0, 1])
p3 = np.array([Lx, 0, Lz])


theta1 = 0*np.pi/180.0
theta2 = 0*np.pi/180.0
theta3 = 0*np.pi/180.0


w = [w1, w2, w3]
p = [p1, p2, p3]
theta = [theta1, theta2, theta3]

M0 = np.array([[1, 0, 0, Lx+Lh], [0, 1, 0, 0], [0, 0, 1, Lz], [0, 0, 0, 1]])

S = ScrewAxesSpace(w, p)
Tsb = ForwardKinematicsSpace_(S, theta, M0)
Js = JacobianSpace(S, theta)
JacobianBody(Tsb, Js)

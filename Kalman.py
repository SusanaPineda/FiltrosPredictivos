import numpy as np


class kalman():
    def __init__(self, A, H, X_ini, P_ini, R, Q, B):
        self.A = A
        self.H = H
        self.X = X_ini
        self.P = P_ini
        self.R = R
        self.Q = Q
        self.B = B


    def predict(self):
        self.X = np.dot(self.A, self.X) + self.B  # u = 0
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.X

    def correct(self, act):
        num = np.dot(self.P, self.H.T)
        den = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        k = np.dot(num, np.linalg.inv(den))
        self.X = np.round(self.X + np.dot(k, (act - np.dot(self.H, self.X))))
        id = np.eye(self.H.shape[1])
        self.P = (id - np.dot(np.dot(k, self.H), self.P))
        return self.X

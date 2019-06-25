"""
This file implements the Extended Kalman Filter.
"""

import numpy as np

from tools.objects import Gaussian
from field_map import FieldMap
from filters.localization_filter import LocalizationFilter
from tools.task import get_motion_noise_covariance
from tools.task import get_observation as get_expected_observation
from tools.task import get_prediction
from tools.task import wrap_angle

def state_jacobian(state, motion):

    x, y, theta = state
    drot1, dtrans, drot2 = motion

    G = np.array([[1, 0, -dtrans * np.sin(theta + drot1)], 
                  [0, 1, dtrans * np.cos(theta + drot1)], 
                  [0, 0, 1]])
    return G

def motion_jacobian(state, motion):

    x, y, theta = state
    drot1, dtrans, drot2 = motion

    V = np.array([[-dtrans * np.sin(theta + drot1), np.cos(theta + drot1), 0],
                  [dtrans * np.cos(theta + drot1),  np.sin(theta + drot1), 0], 
                  [1, 0, 1]])
    return V

def observation_jacobian(state, lm_id):

    x, y, theta = state
    lm_id = int(lm_id)

    field_map = FieldMap()
    mx = field_map.landmarks_poses_x[lm_id]
    my = field_map.landmarks_poses_y[lm_id]
    q = (mx - x)**2 + (my - y)**2

    H = np.array([[(my - y) / q, -(mx - x) / q, -1]])

    return H

class EKF(LocalizationFilter):

    def predict(self, u):
        # TODO Implement here the EKF, perdiction part. HINT: use the auxiliary functions imported above from tools.task

        G = state_jacobian(self.mu, u)
        V = motion_jacobian(self.mu, u)
        mu_bar = get_prediction(self.mu, u)

        #M = get_motion_noise_covariance(u, self._alphas/100) # with reduced motion noise
        M = get_motion_noise_covariance(u, self._alphas) # with standard motion noise

        E_bar = G.dot(self.Sigma).dot(G.T) + V.dot(M).dot(V.T)

        params = Gaussian(mu_bar, E_bar)
        self._state_bar.mu = params.mu
        self._state_bar.Sigma = params.Sigma

    def update(self, z):
        # TODO implement correction step

        obs, _ = get_expected_observation(self.mu_bar, z[1])
        H = observation_jacobian(self.mu_bar, z[1])
        #Q = np.array([[self._Q]])
        
        S = H.dot(self.Sigma_bar).dot(H.T) + self._Q  # with standard observation noise
        #S = H.dot(self.Sigma_bar).dot(H.T) + self._Q/20 # Sensor noise go toward zero
        K = (self.Sigma_bar).dot(H.T).dot(np.linalg.inv(S))
        
        mu = (self._state_bar.mu + K.dot(np.array([[z[0] - obs]])))[:, 0]
        E = (np.eye(3) - K.dot(H)).dot(self.Sigma_bar)

        params = Gaussian(mu, E)
        self._state.mu = params.mu
        self._state.Sigma = params.Sigma

"""
Sudhanva Sreesha
ssreesha@umich.edu
28-Mar-2018

This file implements the Unscented Kalman Filter.
"""

import numpy as np
import scipy

from filters.localization_filter import LocalizationFilter
from tools.task import get_motion_noise_covariance
from tools.task import get_observation
from tools.task import get_prediction
from tools.task import wrap_angle


class UKF(LocalizationFilter):
    def __init__(self, *args, **kwargs):
        super(UKF, self).__init__(*args, **kwargs)

        # TODO add here specific class variables for the UKF

    def predict(self, u):
        # TODO Implement here the UKF, perdiction part
        self._state_bar.mu = self.mu
        self._state_bar.Sigma = self.Sigma

    def update(self, z):
        # TODO implement correction step
        self._state.mu = self._state_bar.mu
        self._state.Sigma = self._state_bar.Sigma

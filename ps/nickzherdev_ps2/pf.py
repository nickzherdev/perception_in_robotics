
"""
Sudhanva Sreesha
ssreesha@umich.edu
28-Mar-2018

This file implements the Particle Filter.
"""

import numpy as np
from numpy.random import uniform
from scipy.stats import norm as gaussian
import random

from filters.localization_filter import LocalizationFilter
from tools.task import get_gaussian_statistics
from tools.task import get_observation
from tools.task import sample_from_odometry
from tools.task import wrap_angle



# https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
def systematic_resample(weights):
    N = len(weights)
    # make N subdivisions, choose positions 
    # with a consistent random offset
    positions = (np.arange(N) + np.random.rand()) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes



class PF(LocalizationFilter):
    def __init__(self, initial_state, alphas, beta, num_particles = 10, global_localization = False):
        super(PF, self).__init__(initial_state, alphas, beta)

        # At initialization we have no reason to favor one particle over another, so we assign a weight of 1/N, for N particles. 
        # We use 1/N so that the sum of all probabilities equals one.

        self.particles = np.ones((num_particles, 3)) * initial_state.mu[:, 0]
        self.num_particles = num_particles
        self.weights_m = np.array([1/num_particles for i in range(self.num_particles)])


    def predict(self, u):
        # TODO Implement here the PF, prediction part
        # Updates mu_bar and Sigma_bar after taking a single prediction step after incorporating the control.
        # :param u: The control for prediction (format: [drot1, dtran, drot2]).

        # we applyed control to every particle and now we have 100 "noisy" particles, each represents x,y,theta


        for i in range(self.num_particles):
            self.particles[i] = sample_from_odometry(self.particles[i], u, self._alphas) 

        params = get_gaussian_statistics(self.particles)
        self._state_bar.mu = params.mu
        self._state_bar.Sigma = params.Sigma

        #print("state_bar", self._state_bar.mu)

    def update(self, z):
        # TODO implement correction step
        # Sequential Importance Sampling, or SIS
        
        # Updates mu and Sigma after incorporating the observation z.
        # :param z: Observation measurement (format: [bearing, marker_id]).
        # we observe one LM at a time
        
        obs = np.zeros(self.num_particles)
        wrapped = []

        for i in range(self.num_particles):
            obs[i], _ = get_observation(self.particles[i], z[1])

        for i, _ in enumerate(obs):
            wrapped.append(wrap_angle(obs[i] - z[0]))

        weights_m = gaussian().pdf(wrapped/np.sqrt(self._Q))
        #weights_m = gaussian(np.mean(wrapped), self._Q).pdf(wrapped)

        self.particles = self.particles[self.low_variance_sampler(weights_m)]
        #self.particles = self.particles[systematic_resample(self.weights_m)]
        
        params = get_gaussian_statistics(self.particles)
        self._state.mu = params.mu
        self._state.Sigma = params.Sigma
        
        #print("state", self._state.mu)

    def low_variance_sampler(self, weights):
        n = self.num_particles
        weigths = np.array(weights) / np.sum(weights)
        indices = []
        C = np.append([], np.cumsum(weigths))
        j = 0
        u0 = (np.random.rand() + np.arange(n)) / n
        for u in u0:
            while j < len(C) and u > C[j]:
                j += 1
            indices += [j - 1]
        return indices
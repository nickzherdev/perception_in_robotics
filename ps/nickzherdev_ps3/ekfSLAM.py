
import numpy as np

from tools.task import get_prediction
from slamBase import SlamBase

from tools.objects import Gaussian
from field_map import FieldMap
#from filters.localization_filter import LocalizationFilter
from tools.task import get_motion_noise_covariance
from tools.task import get_observation as get_expected_observation
from tools.task import get_prediction
from tools.task import wrap_angle
from tools.task import get_invh_prime_wrt_obs, get_invh_prime_wrt_state, get_obs_jcb_wrt_state, get_inverse_observation
from tools.task import get_obs

# partially based on this code
# https://github.com/jfrascon/SLAM_AND_PATH_PLANNING_ALGORITHMS/blob/master/06-SLAM/CODE/slam_09_a_slam_prediction_question.py


class EKF_SLAM(object): # changed this line from SlamBase to object

	def __init__(self, initial_state, Q, alphas):
		# Currently, the number of landmarks is zero.
		self.number_of_landmarks = 0
		self.new_offset = 2*self.number_of_landmarks
		self.known_landmarks = [] # ids

		self.mu = np.array(initial_state.mu)[:, 0]
		self.Sigma = np.array(initial_state.Sigma)

		self.Q = Q
		self.alphas = alphas


	def ekfPrediction(self, u):

		"""
		Updates mu_bar and Sigma_bar after taking a single prediction step after incorporating the control.

		:param u: The control for prediction (format: np.ndarray([drot1, dtran, drot2])).
		:param dt: The time difference between the previous state and the current state being predicted.
		"""
		mu = self.mu.copy()
		Sigma = self.Sigma.copy()

		self.mu[:3] = get_prediction(mu[:3], u)

		G = self.get_g_prime_wrt_state(mu[:3], u)
		GN = np.eye(3 + self.new_offset)
		GN[:3, :3] = G

		V = self.get_g_prime_wrt_motion(mu[:3], u)

		M = get_motion_noise_covariance(u, self.alphas) # with standard motion noise

		RN = np.zeros((3 + self.new_offset, 3 + self.new_offset))
		RN[:3, :3] = V @ M @ V.T

		self.Sigma = GN @ Sigma @ GN.T + RN

		return self.mu, self.Sigma


	def ekfUpdate(self, observations, field_map):

		"""
		Performs data association to figure out previously seen landmarks vs. new landmarks
		in the observations list and updates mu and Sigma after incorporating them.

		:param z: Observation measurements (format: numpy.ndarray of size Kx3
		  observations where each row is [range, bearing, landmark_id]).
		"""

		for obs in observations:
			lm_id = obs[2]

			# check if observed landmark is in list of known lms
			if int(lm_id) not in self.known_landmarks:
				self.initializeNewLandmark(obs)
			else:

				# [x, y, theta, x0, y0, x1, y1, x2, y2].T
				# lm = 0 --> 3, 4
				# lm = 1 --> 5, 6
				# lm = 2 --> 7, 8

				index = 3+2*self.known_landmarks.index(lm_id)

				mu = self.mu.copy()
				Sigma = self.Sigma.copy()

				z = get_obs(mu[:3], mu[index:index + 2])

				h3 = get_obs_jcb_wrt_state(mu[:3], mu[index:index + 2])

				H = np.zeros((2, 3+self.new_offset))
				H[:, :3] = h3

				H[:, index:index + 2] = -h3[:, 0:2]
				S = H @ Sigma @ H.T + self.Q
				K = Sigma @ H.T @ np.linalg.inv(S)

				err = (obs[:2] - z) # range, bearing
				err[1] = wrap_angle(err[1])

				self.mu = mu + (K @ err[:, np.newaxis])[:, 0]
				self.Sigma = (np.eye(3+self.new_offset) - K @ H) @ Sigma

		return self.mu, self.Sigma


	def initializeNewLandmark(self, obs):
		"""    	
		Enlarge the current state and covariance matrix to include one more
		landmark, which is given by its initial_coords (an (x, y) tuple).
		Returns the index of the newly added landmark.
		"""
		rb = obs[:2] # range and bearing
		lm_id = obs[2]

		self.number_of_landmarks += 1
		self.new_offset = 2*self.number_of_landmarks
		self.known_landmarks.append(lm_id)
		Sigma = self.Sigma.copy()
		mu = self.mu.copy()

		L = get_invh_prime_wrt_state(mu[:3], rb)
		W = get_invh_prime_wrt_obs(mu[:3], rb)
		new_lm_coords_wf = get_inverse_observation(mu[:3], rb)
		self.mu = np.concatenate((self.mu, new_lm_coords_wf), axis=None)

		Sigma_m_new = L @ Sigma[:3, :3] @ L.T + W @ self.Q @ W.T   # W.shape (2, 2) - Q.shape (3, 3)
		Sigma_y_new = Sigma[:, :3] @ L.T
		Sigma_new_y = L @ Sigma[:3, :]

		self.Sigma = np.zeros((3 + self.new_offset, 3 + self.new_offset))

		self.Sigma[:-2, :-2] = Sigma
		self.Sigma[-2:, -2:] = Sigma_m_new
		self.Sigma[:-2, -2:] = Sigma_y_new
		self.Sigma[-2:, :-2] = Sigma_new_y


	@staticmethod
	def get_g_prime_wrt_state(state, motion):
		"""
		:param state: The current state mean of the robot (format: np.array([x, y, theta])).
		:param motion: The motion command at the current time step (format: np.array([drot1, dtran, drot2])).
		:return: Jacobian of the state transition matrix w.r.t. the state.
		"""

		drot1, dtran, drot2 = motion

		return np.array([[1, 0, -dtran * np.sin(state[2] + drot1)],
						[0, 1, dtran * np.cos(state[2] + drot1)],
						[0, 0, 1]])


	@staticmethod
	def get_g_prime_wrt_motion(state, motion):
		"""
		:param state: The current state mean of the robot (format: np.array([x, y, theta])).
		:param motion: The motion command at the current time step (format: np.array([drot1, dtran, drot2])).
		:return: Jacobian of the state transition matrix w.r.t. the motion command.
		"""

		drot1, dtran, drot2 = motion

		return np.array([[-dtran * np.sin(state[2] + drot1), np.cos(state[2] + drot1), 0],
						[dtran * np.cos(state[2] + drot1), np.sin(state[2] + drot1), 0],
						[1, 0, 1]])

#!/usr/bin/python

"""
Sudhanva Sreesha
ssreesha@umich.edu
22-Apr-2018

Gonzalo Ferrer
g.ferrer@skoltech.ru
26-Nov-2018
"""

import contextlib
import os
from argparse import ArgumentParser

import numpy as np
from ekfSLAM import EKF_SLAM # added this line
from matplotlib import pyplot as plt
from progress.bar import FillingCirclesBar

from tools.objects import Gaussian
from tools.plot import get_plots_figure
from tools.plot import plot_robot
from field_map import FieldMap
#from slam import SimulationSlamBase
from tools.data import generate_data as generate_input_data
from tools.data import load_data
from tools.plot import plot_field
from tools.plot import plot_observations
from tools.task import get_dummy_context_mgr
from tools.task import get_movie_writer
from tools.plot import plot_landmarks
from tools.plot import plot2dcov


def get_cli_args():
    parser = ArgumentParser('Perception in Robotics PS3')
    parser.add_argument('-i',
                        '--input-data-file',
                        type=str,
                        action='store',
                        help='File with generated data to simulate the filter '
                             'against. Supported format: "npy", and "mat".')
    parser.add_argument('-n',
                        '--num-steps',
                        type=int,
                        action='store',
                        help='The number of time steps to generate data for the simulation. '
                             'This option overrides the data file argument.',
                        default=100)
    parser.add_argument('-f',
                        '--filter',
                        dest='filter_name',
                        choices=['ekf', 'sam'],
                        action='store',
                        help='The slam filter use for the SLAM problem.',
                        default='ekf')
    parser.add_argument('-a',
                        '--alphas',
                        nargs=4,
                        metavar=('A1', 'A2', 'A3', 'A4'),
                        action='store',
                        help='Diagonal of Standard deviations of the Transition noise in action space (M_t).',
                        default=(0.05, 0.001, 0.05, 0.01))
    parser.add_argument('-b',
                        '--beta',
                        nargs=2,
                        metavar=('range', 'bearing (deg)'),
                        action='store',
                        help='Diagonal of Standard deviations of the Observation noise (Q).',
                        default=(10., 10.))
    parser.add_argument('--dt', type=float, action='store', help='Time step (in seconds).', default=0.1)
    parser.add_argument('-s', '--animate', action='store_true', help='Show and animation of the simulation, in real-time.')
    parser.add_argument('--plot-pause-len',
                        type=float,
                        action='store',
                        help='Time (in seconds) to pause the plot animation for between frames.',
                        default=0.01)
    parser.add_argument('--num-landmarks-per-side',
                        type=int,
                        help='The number of landmarks to generate on one side of the field.',
                        default=4)
    parser.add_argument('--max-obs-per-time-step',
                        type=int,
                        help='The maximum number of observations to generate per time step.',
                        default=2)
    parser.add_argument('--data-association',
                        type=str,
                        choices=['known', 'ml', 'jcbb'],
                        default='known',
                        help='The type of data association algorithm to use during the update step.')
    parser.add_argument('--update-type',
                        type=str,
                        choices=['batch', 'sequential'],
                        default='batch',
                        help='Determines how to perform update in the SLAM algorithm.')
    parser.add_argument('-m',
                        '--movie-file',
                        type=str,
                        help='The full path to movie file to write the simulation animation to.',
                        default=None)
    parser.add_argument('--movie-fps',
                        type=float,
                        action='store',
                        help='The FPS rate of the movie to write.',
                        default=10.)
    return parser.parse_args()

def validate_cli_args(args):
    if args.input_data_file and not os.path.exists(args.input_data_file):
        raise OSError('The input data file {} does not exist.'.format(args.input_data_file))

    if not args.input_data_file and not args.num_steps:
        raise RuntimeError('Neither `--input-data-file` nor `--num-steps` were present in the arguments.')


def main():

    args = get_cli_args()
    validate_cli_args(args)
    alphas = np.array(args.alphas)
    beta = np.array(args.beta)


    mean_prior = np.array([180., 50., 0.])
    Sigma_prior = 1e-12 * np.eye(3, 3)
    initial_state = Gaussian(mean_prior, Sigma_prior)

    # Covariance of observation noise. 
    # added this line (block)
    alphas = alphas ** 2
    beta = np.array(beta)
    #print("beta", beta) # beta [10. 10.]
    beta[1] = np.deg2rad(beta[1])
    Q = np.diag([*(beta ** 2), 0])[:2, :2] # wtf 3x3 
    #print("Q", Q)





    if args.input_data_file:
        data = load_data(args.input_data_file)
    elif args.num_steps:
        # Generate data, assuming `--num-steps` was present in the CL args.
        data = generate_input_data(initial_state.mu.T,
                                   args.num_steps,
                                   args.num_landmarks_per_side,
                                   args.max_obs_per_time_step,
                                   alphas,
                                   beta,
                                   args.dt)
    else:
        raise RuntimeError('')

    should_show_plots = True if args.animate else False
    should_write_movie = True if args.movie_file else False
    should_update_plots = True if should_show_plots or should_write_movie else False

    field_map = FieldMap(args.num_landmarks_per_side)

    fig = get_plots_figure(should_show_plots, should_write_movie)
    movie_writer = get_movie_writer(should_write_movie, 'Simulation SLAM', args.movie_fps, args.plot_pause_len)
    progress_bar = FillingCirclesBar('Simulation Progress', max=data.num_steps)



# added
# *************************
    ekf_slam = EKF_SLAM(initial_state, Q, alphas) # added this line

    robot_mu = []
    robot_Sigma = []
    landmarks_mu = []
    landmarks_Sigma = []

    landmarks = ekf_slam.known_landmarks
    n_landmarks = args.num_landmarks_per_side * 2

    landmark_mu = [None] * n_landmarks
    landmark_Sigma = [None] * n_landmarks
# *************************




    with movie_writer.saving(fig, args.movie_file, data.num_steps) if should_write_movie else get_dummy_context_mgr():
        for t in range(data.num_steps):
            # Used as means to include the t-th time-step while plotting.
            tp1 = t + 1

            # Control at the current step.
            u = data.filter.motion_commands[t]
            # Observation at the current step.
            z = data.filter.observations[t]


# added
# *************************
            # TODO SLAM predict(u)
            mu_bar, Sigma_bar = ekf_slam.ekfPrediction(u) 
            
            # TODO SLAM update
            mu, Sigma = ekf_slam.ekfUpdate(z, field_map) 


            for (i, lm) in enumerate(landmarks):
                ind = 2 * i + 3
                landmark_mu[int(lm)] = mu[ind:ind + 2]
                landmark_Sigma[int(lm)] = Sigma[ind:ind + 2, ind:ind + 2]

            robot_mu.append(mu[:3])
            robot_Sigma.append(Sigma[:3, :3])
            landmarks_mu.append(landmark_mu)
            landmarks_Sigma.append(landmark_Sigma)

            ekf_robot_trajectory = np.array(robot_mu)
            ekf_robot_trajectory_Sigma = np.array(robot_Sigma)
            ekf_landmarks_coords = np.array(landmarks_mu)
            ekf_landmarks_cov = np.array(landmarks_Sigma)

# *************************



            progress_bar.next()
            if not should_update_plots:
                continue

            plt.cla()
            plot_field(field_map, z)
            plot_robot(data.debug.real_robot_path[t])
            plot_observations(data.debug.real_robot_path[t],
                              data.debug.noise_free_observations[t],
                              data.filter.observations[t])

            plt.plot(data.debug.real_robot_path[1:tp1, 0], data.debug.real_robot_path[1:tp1, 1], 'm')
            plt.plot(data.debug.noise_free_robot_path[1:tp1, 0], data.debug.noise_free_robot_path[1:tp1, 1], 'g')

            plt.plot([data.debug.real_robot_path[t, 0]], [data.debug.real_robot_path[t, 1]], '*r')
            plt.plot([data.debug.noise_free_robot_path[t, 0]], [data.debug.noise_free_robot_path[t, 1]], '*g')



# added
# *************************
            # TODO plot SLAM solution
            plot_landmarks(ekf_landmarks_coords[t], ekf_landmarks_cov[t])

            plt.plot(ekf_robot_trajectory[1:t + 1, 0], ekf_robot_trajectory[1:t + 1, 1], 'b')

            plot2dcov(ekf_robot_trajectory[t, :2], ekf_robot_trajectory_Sigma[t, :2, :2], 'b')
# *************************





            if should_show_plots:
                # Draw all the plots and pause to create an animation effect.
                plt.draw()
                plt.pause(args.plot_pause_len)

            if should_write_movie:
                movie_writer.grab_frame()

    progress_bar.finish()

    plt.show(block=True)

if __name__ == '__main__':
    main()

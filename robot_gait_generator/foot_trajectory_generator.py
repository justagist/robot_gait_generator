"""
The FootTrajectoryGenerator class generates foot trajectories for a multi-legged robot given gait parameters like
step frequency, duty cycle, phase offset, and foot lift height. It uses either a spline or linear interpolation
to create smooth foot movements.
"""

import numpy as np
from typing import List, Tuple, Literal

from .foot_trajectories import FootTrajectorySpline, FootTrajectoryLinear, FootTrajectory
from .gait_scheduler import GaitScheduler


class FootTrajectoryGenerator:
    """The FootTrajectoryGenerator class generates foot trajectories for a multi-legged robot given gait parameters like
    step frequency, duty cycle, phase offset, and foot lift height. It uses either a spline or linear interpolation
    to create smooth foot movements."""

    def __init__(
        self,
        step_frequencies: float | np.ndarray,
        duty_cycles: float | np.ndarray,
        phase_offsets: np.ndarray,
        foot_lift_height: float,
        relative_feet_targets: np.ndarray | List[np.ndarray],
        foot_start_positions: np.ndarray | List[np.ndarray] | None = None,
        trajectory_type: Literal["linear", "spline"] = "spline",
    ):
        """The FootTrajectoryGenerator class generates foot trajectories for a multi-legged robot given gait parameters
        like step frequency, duty cycle, phase offset, and foot lift height. It uses either a spline or linear
        interpolation to create smooth foot movements.

        Args:
            step_frequencies (float | np.ndarray): Step frequency per foot.
            duty_cycles (float | np.ndarray): Duty cycle of each foot in the gait.
            phase_offsets (np.ndarray): Phase offset for each leg in the gait cycle.
            foot_lift_height (float): Foot lift height for swing phase.
            relative_feet_targets (List[np.ndarray]): End positions of each foot after one gait cycle
                in the world frame wrt the starting positions.
            foot_start_positions (List[np.ndarray], optional): Start positions of each foot in the
                world frame. Defaults to zeros (origin).
            trajectory_type (Literal["linear", "spline"], optional): Type of trajectory to use.
                Defaults to "spline" (4th order spline).
        """

        self.num_legs = len(phase_offsets)

        if foot_start_positions is None:
            foot_start_positions = [np.zeros(3)] * self.num_legs

        foot_start_positions = np.array(foot_start_positions)
        assert foot_start_positions.shape[0] == self.num_legs

        TrajectoryGen: FootTrajectory = (
            FootTrajectorySpline if trajectory_type == "spline" else FootTrajectoryLinear
        )

        self._foot_trajs: List[FootTrajectory] = [
            TrajectoryGen(
                start_point=np.array(foot_start_positions)[i, :],
                end_point=np.array(foot_start_positions)[i, :]
                + np.array(relative_feet_targets)[i, :],
                duty_cycle=duty_cycles[i],
                step_frequency=step_frequencies[i],
                foot_lift_height=foot_lift_height,
            )
            for i in range(self.num_legs)
        ]

        self._scheduler = GaitScheduler(
            step_frequencies=step_frequencies,
            duty_cycles=duty_cycles,
            phase_offsets=phase_offsets,
        )

    @property
    def gait_scheduler(self):
        return self._scheduler

    def get_trajectory_for_duration(
        self, duration: float, timestep: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generates the trajectory for all feet for the specified duration.

        Args:
            duration (float): The duration of the trajectory to generate.
            timestep (float): The timestep for the trajectory.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                A tuple containing the times, positions, velocities, and accelerations of the
                trajectory for all feet.
        """
        num_datapoints = int(duration / timestep) + 1
        times = np.zeros(num_datapoints)
        positions = np.zeros([num_datapoints, self.num_legs, 3])
        velocities = np.zeros([num_datapoints, self.num_legs, 3])
        accelerations = np.zeros([num_datapoints, self.num_legs, 3])
        t = 0.0
        n = 0
        while t <= duration:
            positions[n, :, :], velocities[n, :, :], accelerations[n, :, :] = (
                self.get_foot_trajectory_values_at_time(t=t)
            )
            times[n] = t
            t += timestep
            n += 1

        return times[:n], positions[:n], velocities[:n], accelerations[:n]

    def get_foot_trajectory_values_at_time(
        self, t: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gets the position, velocity, and acceleration of all feet at the specified time.

        Args:
            t (float): The time to get the trajectory values at.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A list containing the position, velocity, and acceleration of
                each foot at the specified time.
        """
        positions = np.zeros([self.num_legs, 3])
        velocities = np.zeros([self.num_legs, 3])
        accelerations = np.zeros([self.num_legs, 3])
        contact_states = self._scheduler.get_contact_states(t)
        phase_completions = self._scheduler.get_phase_completion_percentage(t)
        step_counts = self._scheduler.get_step_count(t)
        for leg_id in range(self.num_legs):
            if contact_states[leg_id] != 1:
                phase_completion = phase_completions[leg_id]
            else:
                phase_completion = 0.0
            positions[leg_id, :] = (
                self._foot_trajs[leg_id].get_trajectory_point_at_phase_fraction(
                    phase_completion=phase_completion
                )
                + step_counts[leg_id] * self._foot_trajs[leg_id].relative_distance_vector
            )
            velocities[leg_id, :] = self._foot_trajs[
                leg_id
            ].get_trajectory_velocity_at_phase_fraction(phase_completion=phase_completion)
            accelerations[leg_id, :] = self._foot_trajs[
                leg_id
            ].get_trajectory_acceleration_at_phase_fraction(phase_completion=phase_completion)
        return positions, velocities, accelerations

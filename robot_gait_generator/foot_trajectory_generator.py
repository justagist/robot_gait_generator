import numpy as np
from typing import List, Tuple

from .foot_trajectory_spline import FootTrajectorySpline
from .gait_scheduler import GaitScheduler


class FootTrajectoryGenerator:
    """Simple foot reference generator that uses FootTrajectorySpline for each leg."""

    def __init__(
        self,
        step_frequencies: float | np.ndarray,
        duty_cycles: float | np.ndarray,
        phase_offsets: np.ndarray,
        foot_lift_height: float,
        end_height: float = 0.0,
        foot_start_positions: List[np.ndarray] = None,
    ):
        """Simple foot trajectory generator that uses FootTrajectorySpline."""

        self.num_legs = len(phase_offsets)

        if foot_start_positions is None:
            foot_start_positions = [np.zeros(3)] * self.num_legs

        foot_start_positions = np.array(foot_start_positions)
        assert foot_start_positions.shape[0] == self.num_legs

        self._foot_splines: List[FootTrajectorySpline] = [
            FootTrajectorySpline(
                start_point=foot_start_positions[i, :],
                end_point=np.array([0.0, 0.0, end_height]),
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

    def get_trajectory_for_duration(
        self, duration: float, timestep: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        num_datapoints = int(duration / timestep) + 1
        times = np.zeros(num_datapoints)
        positions = np.zeros([num_datapoints, self.num_legs, 3])
        velocities = np.zeros([num_datapoints, self.num_legs, 3])
        accelerations = np.zeros([num_datapoints, self.num_legs, 3])
        t = 0.0
        n = 0
        position_offset = np.zeros([self.num_legs, 3])
        while t <= duration:
            for leg_id in range(self.num_legs):
                if self._scheduler.get_contact_states(t)[leg_id] != 1:
                    phase_completion = self._scheduler.get_phase_completion_percentage(
                        t
                    )[leg_id]
                else:
                    phase_completion = 0.0
                    if n > 0:
                        position_offset[leg_id, 2] = positions[n - 1, leg_id, 2]

                positions[n, leg_id, :] = (
                    self._foot_splines[leg_id].get_trajectory_point_at_phase_fraction(
                        phase_completion=phase_completion
                    )
                    + position_offset[leg_id]
                )
                velocities[n, leg_id, :] = self._foot_splines[
                    leg_id
                ].get_trajectory_velocity_at_phase_fraction(
                    phase_completion=phase_completion
                )
                accelerations[n, leg_id, :] = self._foot_splines[
                    leg_id
                ].get_trajectory_acceleration_at_phase_fraction(
                    phase_completion=phase_completion
                )
            times[n] = t

            t += timestep
            n += 1

        return times[:n], positions[:n], velocities[:n], accelerations[:n]

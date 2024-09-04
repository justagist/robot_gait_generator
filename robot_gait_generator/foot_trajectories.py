"""
This file defines classes for generating foot trajectories for legged robots.
It provides the building blocks for generating different types of foot trajectories based on desired gait
parameters, allowing for flexible and customizable robot locomotion.
"""

import numpy as np
from abc import ABC, abstractmethod


class FootTrajectory(ABC):

    def __init__(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        duty_cycle: float,
        step_frequency: float,
        foot_lift_height: float,
    ):
        """A trajectory generator for a single foot given the gait parameters.

        This makes use of the Spline3D class to generate the smooth spline.

        Args:
            start_point (np.ndarray): The first point in the target trajectory.
            end_point (np.ndarray): The end point in the target trajectory.
            duty_cycle (float): Duty cycle. Gait parameter for this leg.
            step_frequency (float): Step frequency. Gait parameter for this leg.
            foot_lift_height (float): Foot lift height.
        """
        self.start_point = start_point
        self.end_point = end_point
        self._duty_cycle = duty_cycle
        self._step_frequency = step_frequency
        self._foot_lift_height = foot_lift_height
        self.spline_duration = (1 - duty_cycle) / step_frequency
        self.relative_distance_vector = self.end_point - self.start_point

    @abstractmethod
    def get_trajectory_point_at_time(self, t: float):
        """Gets the position value on the trajectory at the provided time.

        Args:
            t (float): The desired time.

        Returns:
            np.ndarray: The 3d point on the trajectory.
        """

    @abstractmethod
    def get_trajectory_velocity_at_time(self, t: float) -> np.ndarray:
        """Gets the velocity value on the trajectory at the provided time.

        Args:
            t (float): The desired time.

        Returns:
            np.ndarray: The 3d velocity at time t on the trajectory.
        """

    @abstractmethod
    def get_trajectory_acceleration_at_time(self, t: float) -> np.ndarray:
        """Gets the acceleartion value on the trajectory at the provided time.

        Args:
            t (float): The desired time.

        Returns:
            np.ndarray: The 3d acceleartion at time t on the trajectory.
        """

    def get_trajectory_point_at_phase_fraction(self, phase_completion: float) -> np.ndarray:
        """Gets the position value on the trajectory given the phase completion percentage.

        Args:
            phase_completion (float): The phase completion fraction for the current phase of the
                gait. Value range [0,1].

        Returns:
            np.ndarray: The 3d point on the trajectory.
        """
        return self.get_trajectory_point_at_time(t=phase_completion * self.spline_duration)

    def get_trajectory_velocity_at_phase_fraction(self, phase_completion: float) -> np.ndarray:
        """Gets the velocity value on the trajectory at the provided time.

        Args:
            phase_completion (float): The phase completion fraction for the current phase of the
                gait. Value range [0,1].

        Returns:
            np.ndarray: 3d velocity.
        """
        return self.get_trajectory_velocity_at_time(t=phase_completion * self.spline_duration)

    def get_trajectory_acceleration_at_phase_fraction(self, phase_completion: float) -> np.ndarray:
        """Gets the acceleration value on the trajectory at the provided time.

        Args:
            phase_completion (float): The phase completion fraction for the current phase of the
                gait. Value range [0,1].

        Returns:
            np.ndarray: 3d acceleration.
        """
        return self.get_trajectory_acceleration_at_time(t=phase_completion * self.spline_duration)


class QuarticSpline3D:
    """A 4th order spline between 3 specified waypoints, and respecting 2 velocity conditions."""

    def __init__(self):
        """A 4th order spline between 3 specified waypoints, and respecting 2 velocity conditions."""
        self._coeffs: np.ndarray = None

    def build_spline(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        duration: float,
        mid_point: np.ndarray | None = None,
    ) -> None:
        """Generates the coefficients for the spline.

        This method has to be called before any of the query methods (`get_...()`) can be used.

        Args:
            start_point (np.ndarray): The first point in the target trajectory.
            end_point (np.ndarray): The end point in the target trajectory.
            duration (float): The desired duration for the trajectory (in sec).
            mid_point (np.ndarray, optional): The mid point of the target trajectory. If None
                provided, this method uses the mid point of the start and end point.
        """
        x0, y0, z0 = start_point
        xf, yf, zf = end_point
        if mid_point is None:
            mid_point = (np.array(end_point) + np.array(start_point)) * 0.5
        xm, ym, zm = mid_point

        # redundant variables named for readability of equations
        t0_squared = 0.0
        t0_cubed = 0.0
        t0_pow4 = 0.0

        tf_squared = duration**2
        tf_cubed = duration**3
        tf_pow4 = duration**4

        tm = duration / 2
        tm_squared = tm**2
        tm_cubed = tm**3
        tm_pow4 = tm**4

        # autopep8: off
        # fmt: off
        A_mat = np.array([
            [1, 0, t0_squared, t0_cubed, t0_pow4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, t0_squared, t0_cubed, t0_pow4, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, t0_squared, t0_cubed, t0_pow4],
            [1, tm, tm_squared, tm_cubed, tm_pow4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, tm, tm_squared, tm_cubed, tm_pow4, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, tm, tm_squared, tm_cubed, tm_pow4],
            [1, duration, tf_squared, tf_cubed, tf_pow4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, duration, tf_squared, tf_cubed, tf_pow4, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, duration, tf_squared, tf_cubed, tf_pow4],
            [0, 1, 2 * 0, 3 * t0_squared, 4 * t0_cubed, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 2 * 0, 3 * t0_squared, 4 * t0_cubed, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2 * 0, 3 * t0_squared, 4 * t0_cubed],
            [0, 1, 2 * duration, 3 * tf_squared, 4 * tf_cubed, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 2 * duration, 3 * tf_squared, 4 * tf_cubed, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2 * duration, 3 * tf_squared, 4 * tf_cubed],
        ])
        b_vec = np.array([x0, y0, z0,
                          xm, ym, zm,
                          xf, yf, zf,
                          0,  0,  0,
                          0,  0,  0])
        # fmt: on
        # autopep8: on

        self._coeffs = np.linalg.solve(A_mat, b_vec)

    def _check_built(self):
        """Internal method to check if the spline has been built."""
        if self._coeffs is None:
            raise RuntimeError(
                f"{self.__class__.__name__}: Spline has not been built yet. Call"
                " the `build_spline()` method first, before being able to access `get_point_at_time()`."
            )

    def get_point_at_time(self, t: float) -> np.ndarray:
        """Gets the position value on the trajectory at the provided time.

        Args:
            t (float): The desired time.

        Raises:
            RuntimeError: thrown if the build_splin() method has not been called earlier.

        Returns:
            np.ndarray: The 3d point on the trajectory.
        """
        self._check_built()

        t_squared = t**2
        t_cubed = t**3
        t_pow4 = t**4
        x = (
            self._coeffs[0]
            + self._coeffs[1] * t
            + self._coeffs[2] * t_squared
            + self._coeffs[3] * t_cubed
            + self._coeffs[4] * t_pow4
        )

        y = (
            self._coeffs[5]
            + self._coeffs[6] * t
            + self._coeffs[7] * t_squared
            + self._coeffs[8] * t_cubed
            + self._coeffs[9] * t_pow4
        )

        z = (
            self._coeffs[10]
            + self._coeffs[11] * t
            + self._coeffs[12] * t_squared
            + self._coeffs[13] * t_cubed
            + self._coeffs[14] * t_pow4
        )

        return np.array([x, y, z])

    def get_velocity_at_time(self, t: float) -> np.ndarray:
        """Gets the velocity value on the trajectory at the provided time.

        Args:
            t (float): The desired time.

        Raises:
            RuntimeError: thrown if the build_splin() method has not been called earlier.

        Returns:
            np.ndarray: The 3d velocity at time t on the trajectory.
        """
        self._check_built()

        t_squared = t**2
        t_cubed = t**3

        x = (
            self._coeffs[1]
            + 2 * self._coeffs[2] * t
            + 3 * self._coeffs[3] * t_squared
            + 4 * self._coeffs[4] * t_cubed
        )

        y = (
            self._coeffs[6]
            + 2 * self._coeffs[7] * t
            + 3 * self._coeffs[8] * t_squared
            + 4 * self._coeffs[9] * t_cubed
        )

        z = (
            self._coeffs[11]
            + 2 * self._coeffs[12] * t
            + 3 * self._coeffs[13] * t_squared
            + 4 * self._coeffs[14] * t_cubed
        )

        return np.array([x, y, z])

    def get_acceleration_at_time(self, t: float) -> np.ndarray:
        """Gets the acceleartion value on the trajectory at the provided time.

        Args:
            t (float): The desired time.

        Raises:
            RuntimeError: thrown if the build_splin() method has not been called earlier.

        Returns:
            np.ndarray: The 3d acceleartion at time t on the trajectory.
        """
        self._check_built()

        t_squared = t**2
        x = 2 * self._coeffs[2] + 6 * self._coeffs[3] * t + 12 * self._coeffs[4] * t_squared

        y = 2 * self._coeffs[7] + 6 * self._coeffs[8] * t + 12 * self._coeffs[9] * t_squared

        z = 2 * self._coeffs[12] + 6 * self._coeffs[13] * t + 12 * self._coeffs[14] * t_squared

        return np.array([x, y, z])


class FootTrajectorySpline(FootTrajectory):
    """A trajectory generator for a single foot given the gait parameters. Defines trajectory as a
    4th order spline."""

    def __init__(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        duty_cycle: float,
        step_frequency: float,
        foot_lift_height: float,
    ):
        """A trajectory generator for a single foot given the gait parameters.

        This makes use of the Spline3D class to generate the smooth spline.

        Args:
            start_point (np.ndarray): The first point in the target trajectory.
            end_point (np.ndarray): The end point in the target trajectory.
            duty_cycle (float): Duty cycle. Gait parameter for this leg.
            step_frequency (float): Step frequency. Gait parameter for this leg.
            foot_lift_height (float): Foot lift height.
        """
        super().__init__(
            start_point=start_point,
            end_point=end_point,
            duty_cycle=duty_cycle,
            step_frequency=step_frequency,
            foot_lift_height=foot_lift_height,
        )
        self._spline = QuarticSpline3D()

        # use the foot height and the midpoint between start and end (along x and y) as the midpoint
        spline_midpoint = (np.array(self.end_point) + np.array(self.start_point)) * 0.5
        spline_midpoint[2] += self._foot_lift_height

        self._spline.build_spline(
            start_point=self.start_point,
            end_point=self.end_point,
            duration=self.spline_duration,
            mid_point=spline_midpoint,
        )

    def get_trajectory_point_at_time(self, t: float):
        """Gets the position value on the trajectory at the provided time.

        Args:
            t (float): The desired time.

        Returns:
            np.ndarray: The 3d point on the trajectory.
        """
        if t < 0:
            return self.start_point
        if t >= self.spline_duration:
            return self.end_point
        return self._spline.get_point_at_time(t=t)

    def get_trajectory_velocity_at_time(self, t: float) -> np.ndarray:
        """Gets the velocity value on the trajectory at the provided time.

        Args:
            t (float): The desired time.

        Returns:
            np.ndarray: The 3d velocity at time t on the trajectory.
        """
        if t < 0 or t > self.spline_duration:
            return np.zeros(3)
        return self._spline.get_velocity_at_time(t=t)

    def get_trajectory_acceleration_at_time(self, t: float) -> np.ndarray:
        """Gets the acceleartion value on the trajectory at the provided time.

        Args:
            t (float): The desired time.

        Returns:
            np.ndarray: The 3d acceleartion at time t on the trajectory.
        """
        if t < 0 or t > self.spline_duration:
            return np.zeros(3)
        return self._spline.get_acceleration_at_time(t=t)


class FootTrajectoryLinear(FootTrajectory):
    """A trajectory generator for a single foot given the gait parameters. Defines trajectory as a
    linear interpolation.
    """

    def get_trajectory_point_at_time(self, t: float) -> np.ndarray:
        if t < 0:
            return self.start_point
        if t >= self.spline_duration:
            return self.end_point
        dist_vect = self.start_point + (self.relative_distance_vector) * t / self.spline_duration
        dist_vect[2] += (1 - (abs(0.5 - t / self.spline_duration) / 0.5)) * self._foot_lift_height
        return dist_vect

    def get_trajectory_velocity_at_time(self, t: float) -> np.ndarray:
        if t < 0 or t >= self.spline_duration:
            return np.zeros(3)
        vel = (self.relative_distance_vector) / self.spline_duration
        if t < self.spline_duration / 2:
            vel[2] += (self._foot_lift_height) / (self.spline_duration / 2.0)
        else:
            vel[2] -= (self._foot_lift_height) / (self.spline_duration / 2.0)
        return vel

    def get_trajectory_acceleration_at_time(self, t: float) -> np.ndarray:
        return np.zeros(3)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    start_point = np.array([0.0, 0.0, 0.0])
    end_point = np.array([0.2, 0.0, 0.0])
    duty_cycle = 0.5
    step_frequency = 2.0
    foot_lift_height = 0.1

    # Create a FootTrajectoryLinear object
    trajectory = FootTrajectorySpline(
        start_point=start_point,
        end_point=end_point,
        duty_cycle=duty_cycle,
        step_frequency=step_frequency,
        foot_lift_height=foot_lift_height,
    )

    # Generate trajectory points
    times = np.linspace(0, 0.5, 100)
    positions = np.zeros((len(times), 3))
    velocities = np.zeros((len(times), 3))
    accelerations = np.zeros((len(times), 3))

    for i, t in enumerate(times):
        positions[i, :] = trajectory.get_trajectory_point_at_time(t)
        velocities[i, :] = trajectory.get_trajectory_velocity_at_time(t)
        accelerations[i, :] = trajectory.get_trajectory_acceleration_at_time(t)

    # Plot the trajectory
    fig, axs = plt.subplots(3, 1, sharex=True)
    axs[0].plot(times, positions[:, 0], label="x")
    axs[0].plot(times, positions[:, 1], label="y")
    axs[0].plot(times, positions[:, 2], label="z")
    axs[0].set_ylabel("Position")
    axs[0].legend()

    axs[1].plot(times, velocities[:, 0], label="x")
    axs[1].plot(times, velocities[:, 1], label="y")
    axs[1].plot(times, velocities[:, 2], label="z")
    axs[1].set_ylabel("Velocity")
    axs[1].legend()

    axs[2].plot(times, accelerations[:, 0], label="x")
    axs[2].plot(times, accelerations[:, 1], label="y")
    axs[2].plot(times, accelerations[:, 2], label="z")
    axs[2].set_ylabel("Acceleration")
    axs[2].legend()

    plt.xlabel("Time (s)")
    plt.show()

    print(positions[-10:, 0])

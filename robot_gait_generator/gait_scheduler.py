import numpy as np
from numbers import Number


class GaitScheduler:
    """Generate gait schedule for multiple feet."""

    step_frequencies: np.ndarray
    duty_cycles: np.ndarray
    phase_offsets: np.ndarray
    step_durations: np.ndarray
    stance_durations: np.ndarray
    swing_durations: np.ndarray
    step_frequencies: np.ndarray

    def __init__(
        self,
        step_frequencies: float | np.ndarray = 1.0,
        duty_cycles: float | np.ndarray = 0.5,
        phase_offsets: np.ndarray = 0.0,
    ):
        """Constructor.

        Generate gait schedule for multiple feet.

        Args:
            step_frequencies (float, optional): step_frequencies = 1/T, T is step duration. Defaults to 1.0.
            duty_cycles (float, optional): range [0, 1], duty_cycles = on_time/(on_time + off_time). Defaults to 0.5.
            phase_offsets (float, optional): range [0, 1]. Offsets of each foot's gait cycle Defaults to 0.0.
        """
        if isinstance(step_frequencies, Number):
            step_frequencies = [step_frequencies]
        if isinstance(duty_cycles, Number):
            duty_cycles = [duty_cycles]
        if isinstance(phase_offsets, Number):
            phase_offsets = [phase_offsets]

        assert len(phase_offsets) == len(duty_cycles) == len(step_frequencies)

        self.step_frequencies = np.array(step_frequencies)
        self.duty_cycles = np.array(duty_cycles)
        self.phase_offsets = np.array(phase_offsets)
        self._update_durations(self.step_frequencies, self.duty_cycles)
        # self._prev_contact_states = self.get_contact_states(0)

    def _update_durations(
        self, step_frequencies: float | np.ndarray, duty_cycles: float | np.ndarray
    ):
        self.step_durations = 1 / step_frequencies
        self.stance_durations = self.step_durations * duty_cycles
        self.swing_durations = self.step_durations * (1 - duty_cycles)

    def reset(
        self,
        step_frequencies: float | np.ndarray | None = None,
        duty_cycles: float | np.ndarray | None = None,
        phase_offsets: float | np.ndarray | None = None,
    ):
        if step_frequencies is not None:
            self.step_frequencies = np.array(step_frequencies)
        if duty_cycles is not None:
            self.duty_cycles = np.array(duty_cycles)
        if phase_offsets is not None:
            self.phase_offsets = np.array(phase_offsets)
        self._update_durations(self.step_frequencies, self.duty_cycles)

    def get_theta(self, t: float):
        return self.step_frequencies * t + self.phase_offsets

    def get_step_count(self, t: float):
        return np.array(self.get_theta(t) // 1 - self.get_theta(0) // 1).astype(int)

    def get_step_completion(self, t: float):
        return self.get_theta(t) % 1

    def get_contact_states(self, t: float):
        return np.array(self.get_step_completion(t) < self.duty_cycles).astype(int)

    def get_phase_completion_percentage(self, t: float):
        return self.get_step_completion(t) / self.duty_cycles * self.get_contact_states(t) + (
            self.get_step_completion(t) - self.duty_cycles
        ) / (1 - self.duty_cycles) * (1 - self.get_contact_states(t))

    def get_phase_duration(self, t: float):
        return self.stance_durations * self.get_contact_states(t) + self.swing_durations * (
            1 - self.get_contact_states(t)
        )

    def get_phase_time_past(self, t: float):
        return self.get_phase_completion_percentage(t) * self.get_phase_duration(t)

    def get_phase_time_left(self, t: float):
        return self.get_phase_duration(t) - self.get_phase_time_past(t)

    def get_phase_starting_time(self, t: float):
        return t - self.get_phase_time_past(t)

    def get_phase_ending_time(self, t: float):
        return t + self.get_phase_time_left(t)

    def get_touchdown_moments(self, cycle_index: int):
        return self.step_durations * (np.array(cycle_index) - self.phase_offsets)

    def get_takeoff_moments(self, cycle_index: int):
        return self.get_touchdown_moments(cycle_index) - self.swing_durations

    def get_contact_switch_times_matrix(self, start_t: float, end_t: float) -> np.ndarray:
        """Generate a matrix that returns contact state per feet/ee and timestamp for
        each time contact state for an ee changes within the specified duration.
        """
        t = start_t
        prev_vals = np.ones(len(self.duty_cycles)) * -1
        out = []
        while t <= end_t:
            vals = self.get_contact_states(t)
            if np.any(vals != prev_vals):
                out.append([t, *vals])
            t += 0.001
            prev_vals = vals.copy()
        return np.array(out)


if __name__ == "__main__":
    scheduler = GaitScheduler(
        step_frequencies=[1] * 4,
        duty_cycles=[0.55] * 4,
        phase_offsets=[0, 0.5, 0.5, 0],
    )
    print(scheduler.get_contact_switch_times_matrix(0, 5))
    # t = 0.0
    # prev_vals = np.array([-1, -1, -1, -1])
    # out = []
    # while t <= 10:
    #     vals = scheduler.get_contact_states(t)
    #     if np.all(vals != prev_vals):
    #         print(t, vals)
    #         out.append([t, *vals])
    #     t += 0.1
    #     prev_vals = vals.copy()

    # print(repr(np.array(out)))

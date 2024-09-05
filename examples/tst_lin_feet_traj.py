import numpy as np
import matplotlib.pyplot as plt

from robot_gait_generator.foot_trajectory_generator import FootTrajectoryGenerator

if __name__ == "__main__":

    # Define gait parameters
    step_frequencies = np.array([1.0, 1.0, 1.0, 1.0])  # Steps per second per leg
    duty_cycles = np.array([0.5, 0.5, 0.5, 0.5])  # Percentage of gait cycle in stance
    phase_offsets = np.array([0.15, 0.0, 0.0, 0.0])  # Phase offset of each leg
    foot_lift_height = 0.1  # Foot lift height during swing phase
    relative_feet_targets = [
        [
            0.2,
            0.0,
            0.0,
        ]
    ] * 4  # Foot height at the end of the swing phase (touchdown)

    # Create a FootTrajectoryGenerator object
    generator = FootTrajectoryGenerator(
        step_frequencies=step_frequencies,
        duty_cycles=duty_cycles,
        phase_offsets=phase_offsets,
        foot_lift_height=foot_lift_height,
        relative_feet_targets=relative_feet_targets,
        trajectory_type="linear",
    )

    # Generate trajectory for a specific duration
    duration = 5.0
    timestep = 0.01
    times, positions, velocities, accelerations = generator.get_trajectory_for_duration(
        duration=duration, timestep=timestep
    )

    # Plot the trajectory for each foot
    fig, axs = plt.subplots(4, 1, sharex=True)
    for leg_id in range(4):
        axs[leg_id].plot(times, positions[:, leg_id, 0], label=f"Leg {leg_id} X")
        axs[leg_id].plot(times, positions[:, leg_id, 2], label=f"Leg {leg_id} Z")
        axs[leg_id].set_ylabel("Foot Height (m)")
        axs[leg_id].legend()

    plt.xlabel("X")
    plt.suptitle("Foot Trajectories")
    plt.show()

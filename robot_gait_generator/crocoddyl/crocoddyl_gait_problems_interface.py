"""
This class provides a convenient interface for setting up generic gait generation problems in Crocoddyl. It handles
the initialization of the robot model, foot positions, and CoM, and then leverages the create_generic_gait_models
method to generate the actual gait model sequence. Finally, it constructs a crocoddyl.ShootingProblem that can be
solved using Crocoddyl's optimization algorithms to find the optimal trajectory for the robot to achieve the desired
gait.

This class provides a framework for easily defining and generating complex gait patterns for robots using Crocoddyl.
It handles the low-level details of creating action models, contact models, and cost functions, allowing users to
focus on higher-level gait parameters, and provides a the fully defined problem that can be directly solved using
solvers in Crocoddyl.
"""

from typing import List
import numpy as np
import pinocchio

from .crocoddyl_gait_model_interface import (
    Vector3D,
    CrocoddylGaitModelInterface,
)
import crocoddyl

# pylint: disable = E1101, E0401, W0511


class CrocoddylGaitProblemsInterface(CrocoddylGaitModelInterface):
    """Helper class to create generic crocoddyl gait models that can be used
    for defining gait problems in crocoddyl.
    This class aims to help build simple locomotion problems.
    """

    def create_generic_gait_problem(
        self,
        x0: np.ndarray,
        step_frequencies: float | np.ndarray,
        duty_cycles: float | np.ndarray,
        phase_offsets: np.ndarray,
        relative_feet_targets: List[Vector3D],
        starting_feet_heights: List[float],
        foot_lift_height: float | List[float],
        duration: float,
        time_step: float,
    ) -> "crocoddyl.ShootingProblem":
        """Create a shooting problem by generating gait model sequence given
        the gait parameters and duration of motion.

        Args:
            x0 (np.ndarray): Initial state (np.concatenate(q,v))
            step_frequencies (float | np.ndarray): Step frequency per foot. It refers to how many steps a leg
                takes per second. It essentially dictates the cadence or tempo of a leg's movement during locomotion.
            duty_cycles (float | np.ndarray): Duty cycle of each foot in the gait.
                It refers to the percentage of time a leg spends in the stance phase (in contact with the
                ground) within a single, complete gait cycle.
            phase_offsets (np.ndarray): Phase offset for each leg in the gait cycle.
                Determines the relative timing of each leg's step cycle within the overall gait.
                Imagine each leg having its own independent clock controlling its swing and stance phases.
                Phase offsets are like adjusting the starting time on these clocks.
            relative_feet_targets (List[Vector3D]): Relative foot target (stride lengths)
            starting_feet_heights (List[float]): Initial height of each foot.
            foot_lift_height (float | List[float]): Foot lift height for swing phase.
            duration (float): Total duration of the motion.
            time_step (float): Time step for discretising the models.

        Returns:
            crocoddyl.ShootingProblem: Shooting problem defining the full locomotion task.
        """
        q0 = x0[: self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)

        init_feet_pos = [self.rdata.oMf[int(f_id)].translation for f_id in self._ee_ids]
        for ee_idx in range(len(init_feet_pos)):
            init_feet_pos[ee_idx][2] = starting_feet_heights[ee_idx]
        init_com_pos = np.mean(init_feet_pos, axis=0)
        # CoM trajectory is always initialised to be at the midpoint of the feet
        # except height (which is obtained from the pincchio model)
        init_com_pos[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2].item()

        gait_models = self.create_generic_gait_models(
            init_com_pos=init_com_pos,
            init_feet_pos=init_feet_pos,
            step_frequencies=step_frequencies,
            duty_cycles=duty_cycles,
            phase_offsets=phase_offsets,
            relative_feet_targets=relative_feet_targets,
            foot_lift_height=foot_lift_height,
            duration=duration,
            time_step=time_step,
        )

        problem = crocoddyl.ShootingProblem(x0, gait_models[:-1], gait_models[-1])
        return problem

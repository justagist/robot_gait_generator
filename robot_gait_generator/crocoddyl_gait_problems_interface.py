"""
TODO:
    - [ ] refactor to use foot trajectory spline
    - [ ] refactor CoM trajectory generator
"""

from typing import List, Literal
import numpy as np
import pinocchio

from .crocoddyl_gait_model_interface import (
    Vector3D,
    CrocoddylGaitModelInterface,
)
import crocoddyl

# pylint: disable = E1101, E0401, W0511


class CrocoddylGaitProblemsInterface(CrocoddylGaitModelInterface):

    def __init__(
        self,
        pinocchio_robot_model: pinocchio.Model,
        ee_names: List[str],
        default_standing_configuration: np.ndarray,
        integrator: Literal["euler", "rk4"] = "euler",
        control: Literal["zero", "one", "rk4"] = "zero",
        fwddyn: bool = True,
        mu: float = 0.7,
        use_pseudo_impulse_model: bool = False,
    ):
        super().__init__(
            pinocchio_robot_model=pinocchio_robot_model,
            ee_names=ee_names,
            default_standing_configuration=default_standing_configuration,
            integrator=integrator,
            control=control,
            fwddyn=fwddyn,
            mu=mu,
            use_pseudo_impulse_model=use_pseudo_impulse_model,
        )

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
            step_frequencies (float | np.ndarray): Step frequency per foot.
            duty_cycles (float | np.ndarray): Duty cycle of each foot in the gait.
            phase_offsets (np.ndarray): Phase offset for each leg in the gait cycle.
            relative_feet_targets (List[Vector3D]): Relative foot target (stride lengths)
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
        # except height
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

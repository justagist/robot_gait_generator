"""
This code provides a framework for easily defining and generating complex gait patterns for robots using Crocoddyl.
It handles the low-level details of creating action models, contact models, and cost functions, allowing users to
focus on higher-level gait parameters.
"""

# pylint: disable = E1101, E0401, W0511

from typing import List, Literal, Tuple, TypeAlias, Union
import numpy as np
import pinocchio
import crocoddyl
from ..foot_trajectory_generator import FootTrajectoryGenerator

Vector3D: TypeAlias = np.ndarray
"""Numpy array representating 3D cartesian vector in format [x,y,z]"""


class CrocoddylGaitModelInterface:
    """Helper class to create composable crocoddyl gait models that can be used
    for defining gait problems in crocoddyl.
    This class aims to help build simple locomotion problems."""

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
        """Helper class to create composable crocoddyl gait models that can be used
        for defining gait problems in crocoddyl.
        This class aims to help build simple locomotion problems.

        This class aims to help build simple locomotion problems.

        Args:
            pinocchio_robot_model (pinocchio.Model): Pinocchio model object describing the
                dynamics of this robot.
            ee_names (List[str]): Names of end-effectors of this robot.
            default_standing_configuration (np.ndarray): Pinocchio configuration (q) vector
                that is to be used as the standing configuration for this robot.
            integrator (Literal["euler", "rk4"]], optional): type of integrator. Defaults to "euler".
            control (Literal["zero", "one", "rk4"], optional): type of control parametrization.
                Defaults to "zero".
            fwddyn (bool, optional): True for forward-dynamics and False for inverse-dynamics
                formulations. Defaults to True.
            mu (float, optional): Coefficient of friction for contacts. Defaults to 0.7.
        """

        self.rmodel = pinocchio_robot_model

        self._ee_ids = np.array([self.rmodel.getFrameId(ee_name) for ee_name in ee_names])
        self._ee_name_to_id = dict(zip(ee_names, self._ee_ids.tolist()))
        self._integrator = integrator
        self._control = control
        self._fwddyn = fwddyn
        self._mu = mu

        self.rdata = self.rmodel.createData()
        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)

        # Defining default state
        self.rmodel.defaultState = np.concatenate(
            [default_standing_configuration, np.zeros(self.rmodel.nv)]
        )
        self._r_surf = np.eye(3)
        self._use_pseudo_impulse_model = use_pseudo_impulse_model

    def get_ee_id(self, ee_name: str):
        return self._ee_name_to_id[ee_name]

    def create_generic_gait_models(
        self,
        init_com_pos: Vector3D,
        init_feet_pos: List[Vector3D],
        step_frequencies: float | np.ndarray,
        duty_cycles: float | np.ndarray,
        phase_offsets: np.ndarray,
        relative_feet_targets: List[Vector3D],
        foot_lift_height: float | List[float],
        duration: float,
        time_step: float,
        feet_trajectory_type: Literal["linear", "spline"] = "spline",
    ) -> List[crocoddyl.IntegratedActionModelAbstract]:
        """Create a full (generic) gait model sequence given gait parameters
        and duration of motion.

        Args:
            init_com_pos (Vector3D): Initial position of CoM in the world.
            init_feet_pos (List[Vector3D]): Initial feet positions in the world.
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
            foot_lift_height (float | List[float]): Foot lift height for swing phase.
            duration (float): Total duration of the motion.
            time_step (float): Time step for discretising the models.

        Returns:
            List[crocoddyl.IntegratedActionModelAbstract]: Sequence of gait models
                that can be used to formulate the shooting problem.
        """

        traj_gen = FootTrajectoryGenerator(
            step_frequencies=step_frequencies,
            duty_cycles=duty_cycles,
            phase_offsets=phase_offsets,
            foot_lift_height=foot_lift_height,
            relative_feet_targets=relative_feet_targets,
            foot_start_positions=init_feet_pos,
            trajectory_type=feet_trajectory_type,
        )
        relative_feet_targets = np.array(relative_feet_targets)
        contact_event_matrix = traj_gen.gait_scheduler.get_contact_switch_times_matrix(
            start_t=0.0, end_t=duration
        )

        event_id = 0
        t = 0.0
        assert contact_event_matrix[0, 0] == t

        assert (
            len(init_feet_pos)
            == relative_feet_targets.shape[0]
            == len(traj_gen.gait_scheduler.swing_durations)
        )
        dp = np.zeros([len(init_feet_pos), 3])
        for i in range(len(init_feet_pos)):
            dp[i, :] = (
                relative_feet_targets[i, :] * time_step / traj_gen.gait_scheduler.swing_durations[i]
            )

        stance_feet_col_ids = np.nonzero(contact_event_matrix[event_id, 1:])[0]
        swing_feet_col_ids = np.where(contact_event_matrix[event_id, 1:] == 0)[0]
        swing_foot_task: List[Tuple[int, pinocchio.SE3, Vector3D]] = []
        # NOTE: this CoM percentage is how it is done in the crocoddyl examples (should find
        # a better way to use a generic CoM planner)
        com_percentage = len(swing_feet_col_ids) / len(self._ee_ids)
        com_pos = np.array(init_com_pos)

        models: List[crocoddyl.IntegratedActionModelAbstract] = []

        while t <= duration:
            if (
                event_id < contact_event_matrix.shape[0] - 1
                and contact_event_matrix[event_id + 1, 0] <= t
            ):
                event_id += 1
                stance_feet_col_ids = np.nonzero(contact_event_matrix[event_id, 1:])[0]
                swing_feet_col_ids = np.where(contact_event_matrix[event_id, 1:] == 0)[0]
                com_percentage = len(swing_feet_col_ids) / len(self._ee_ids)
                models += [
                    self.create_foot_switch_model(
                        support_foot_ids=self._ee_ids[stance_feet_col_ids],
                        swing_foot_task=swing_foot_task,
                        pseudo_impulse=self._use_pseudo_impulse_model,
                    )
                ]

            if len(swing_feet_col_ids) > 0:
                com_pos += np.mean(dp[swing_feet_col_ids, :], axis=0) * com_percentage
            swing_foot_task: List[Tuple[int, pinocchio.SE3, Vector3D]] = []
            foot_targets, _, _ = traj_gen.get_foot_trajectory_values_at_time(t=t)
            for col_id in swing_feet_col_ids:
                swing_foot_task += [
                    [
                        int(self._ee_ids[col_id]),
                        pinocchio.SE3(np.eye(3), foot_targets[col_id, :]),
                    ]
                ]
            models += [
                self.create_swing_foot_model(
                    time_step=time_step,
                    support_foot_ids=self._ee_ids[stance_feet_col_ids],
                    com_task=com_pos,
                    swing_foot_task=swing_foot_task,
                )
            ]
            t += time_step
        return models

    def create_swing_foot_model(
        self,
        time_step: float,
        support_foot_ids: List[int],
        com_task: Vector3D = None,
        swing_foot_task: List[Tuple[int, pinocchio.SE3, Vector3D]] = None,
    ) -> crocoddyl.IntegratedActionModelAbstract:
        """Action model for a swing foot phase.

        Args:
            time_step (float): step duration of the action model
            support_foot_ids (List[int]): Ids of the constrained feet
            com_task (Vector3D, optional): CoM task. Defaults to None.
            swing_foot_task (List[Tuple[int, pinocchio.SE3, Vector3D]], optional): swinging
                foot task. Defaults to None.

        Returns:
            crocoddyl.IntegratedActionModelAbstract: action model for a swing foot phase
        """
        # NOTE: crocoddyl formulates a dynamics model as a function of current state (x),
        # actuation (floating base speeds/efforts), contact models, and cost models

        # Creating a 3D multi-contact model, and then including the supporting foot
        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 3 * len(support_foot_ids)
        # TODO: check why friction constraint is not already included in ContactModelMultiple
        # TODO: Understand difference b/w contact models and cost models
        contact_model = crocoddyl.ContactModelMultiple(self.state, nu)
        for i in support_foot_ids:
            support_contact_model = crocoddyl.ContactModel3D(
                self.state,
                int(i),
                np.array([0.0, 0.0, 0.0]),
                pinocchio.LOCAL_WORLD_ALIGNED,
                nu,
                np.array([0.0, 50.0]),
            )
            # contact constraint for the support feet
            contact_model.addContact(
                self.rmodel.frames[int(i)].name + "_contact", support_contact_model
            )

        # Creating the cost model for a contact phase
        cost_model = crocoddyl.CostModelSum(self.state, nu)
        if isinstance(com_task, np.ndarray):
            com_residual = crocoddyl.ResidualModelCoMPosition(self.state, com_task, nu)
            com_track = crocoddyl.CostModelResidual(self.state, com_residual)
            cost_model.addCost("com_track", com_track, 1e6)
        for i in support_foot_ids:
            cone = crocoddyl.FrictionCone(self._r_surf, self._mu, 4, False)
            cone_residual = crocoddyl.ResidualModelContactFrictionCone(
                self.state, int(i), cone, nu, self._fwddyn
            )
            cone_activation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
            )
            friction_cone = crocoddyl.CostModelResidual(self.state, cone_activation, cone_residual)
            # friction cone constraint for the support feet
            cost_model.addCost(
                self.rmodel.frames[int(i)].name + "_frictionCone", friction_cone, 1e1
            )
        if swing_foot_task is not None:
            for i in swing_foot_task:
                # NOTE: this is purely a position tracking task. NO ORIENTATION TRACKING
                # FOR SWING TASK
                frame_translation_residual = crocoddyl.ResidualModelFrameTranslation(
                    self.state, i[0], i[1].translation, nu
                )
                # TODO: read about cost model residuals
                foot_track = crocoddyl.CostModelResidual(self.state, frame_translation_residual)
                cost_model.addCost(self.rmodel.frames[i[0]].name + "_footTrack", foot_track, 1e6)
        # TODO: verify if this is weightage for position or vel/acc/torque
        state_weights = np.array(
            [0.0] * 3
            + [500.0] * 3
            + [0.01] * (self.rmodel.nv - 6)
            + [10.0] * 6
            + [1.0] * (self.rmodel.nv - 6)
        )
        state_residual = crocoddyl.ResidualModelState(self.state, self.rmodel.defaultState, nu)
        state_activation = crocoddyl.ActivationModelWeightedQuad(state_weights**2)
        state_reg = crocoddyl.CostModelResidual(self.state, state_activation, state_residual)
        if self._fwddyn:
            ctrl_residual = crocoddyl.ResidualModelControl(self.state, nu)
            ctrl_reg = crocoddyl.CostModelResidual(self.state, ctrl_residual)
        else:
            ctrl_residual = crocoddyl.ResidualModelJointEffort(self.state, self.actuation, nu)
            ctrl_reg = crocoddyl.CostModelResidual(self.state, ctrl_residual)
        cost_model.addCost("state_reg", state_reg, 1e1)
        cost_model.addCost("ctrl_reg", ctrl_reg, 1e-1)

        lb = np.concatenate([self.state.lb[1 : self.state.nv + 1], self.state.lb[-self.state.nv :]])
        ub = np.concatenate([self.state.ub[1 : self.state.nv + 1], self.state.ub[-self.state.nv :]])
        state_bounds_residual = crocoddyl.ResidualModelState(self.state, nu)
        state_bounds_activation = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(lb, ub)
        )
        state_bounds = crocoddyl.CostModelResidual(
            self.state, state_bounds_activation, state_bounds_residual
        )
        cost_model.addCost("state_bounds", state_bounds, 1e3)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        if self._fwddyn:
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state, self.actuation, contact_model, cost_model, 0.0, True
            )
        else:
            dmodel = crocoddyl.DifferentialActionModelContactInvDynamics(
                self.state, self.actuation, contact_model, cost_model
            )
        if self._control == "one":
            control = crocoddyl.ControlParametrizationModelPolyOne(nu)
        elif self._control == "rk4":
            control = crocoddyl.ControlParametrizationModelPolyTwoRK(nu, crocoddyl.RKType.four)
        elif self._control == "rk3":
            control = crocoddyl.ControlParametrizationModelPolyTwoRK(nu, crocoddyl.RKType.three)
        else:
            control = crocoddyl.ControlParametrizationModelPolyZero(nu)
        if self._integrator == "euler":
            model = crocoddyl.IntegratedActionModelEuler(dmodel, control, time_step)
        elif self._integrator == "rk4":
            model = crocoddyl.IntegratedActionModelRK4(dmodel, control, time_step)
        return model

    def create_foot_switch_model(
        self,
        support_foot_ids: List[int],
        swing_foot_task: List[Tuple[int, pinocchio.SE3, Vector3D]],
        pseudo_impulse: bool = False,
    ) -> Union[
        crocoddyl.IntegratedActionModelEuler,
        crocoddyl.IntegratedActionModelRK,
        crocoddyl.ActionModelImpulseFwdDynamics,
    ]:
        """Action model for a foot switch phase.

        Args:
            support_foot_ids (List[int]): Ids of the constrained feet (stance feet)
            swing_foot_task (List[Tuple[int, pinocchio.SE3, Vector3D]]): swinging foot tasks
            pseudo_impulse (bool, optional): true for pseudo-impulse models, otherwise it uses the
                impulse model. Defaults to False.

        Returns:
            (crocoddyl.IntegratedActionModelEuler | crocoddyl.IntegratedActionModelRK| crocoddyl.ActionModelImpulseFwdDynamics): action model for a foot switch phase
        """
        if pseudo_impulse:
            return self.create_pseudo_impulse_model(support_foot_ids, swing_foot_task)
        else:
            return self.create_impulse_model(support_foot_ids, swing_foot_task)

    def create_pseudo_impulse_model(
        self,
        support_foot_ids: List[int],
        swing_foot_task: List[Tuple[int, pinocchio.SE3, Vector3D]],
    ) -> Union[crocoddyl.IntegratedActionModelEuler, crocoddyl.IntegratedActionModelRK]:
        """Action model for pseudo-impulse models.

        A pseudo-impulse model consists of adding high-penalty cost for the contact
        velocities.

        Args:
            support_foot_ids (List[int]): Ids of the constrained feet (stance feet)
            swing_foot_task (List[Tuple[int, pinocchio.SE3, Vector3D]]): swinging foot tasks

        Returns:
            crocoddyl.IntegratedActionModelEuler | crocoddyl.IntegratedActionModelRK: pseudo-impulse
                differential action model
        """
        # Creating a 3D multi-contact model, and then including the supporting
        # foot
        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 3 * len(support_foot_ids)
        contact_model = crocoddyl.ContactModelMultiple(self.state, nu)
        for i in support_foot_ids:
            support_contact_model = crocoddyl.ContactModel3D(
                self.state,
                int(i),
                np.array([0.0, 0.0, 0.0]),
                pinocchio.LOCAL_WORLD_ALIGNED,
                nu,
                np.array([0.0, 50.0]),
            )
            contact_model.addContact(
                self.rmodel.frames[int(i)].name + "_contact", support_contact_model
            )

        # Creating the cost model for a contact phase
        cost_model = crocoddyl.CostModelSum(self.state, nu)
        for i in support_foot_ids:
            cone = crocoddyl.FrictionCone(self._r_surf, self._mu, 4, False)
            cone_residual = crocoddyl.ResidualModelContactFrictionCone(
                self.state, int(i), cone, nu, self._fwddyn
            )
            cone_activation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
            )
            friction_cone = crocoddyl.CostModelResidual(self.state, cone_activation, cone_residual)
            cost_model.addCost(
                self.rmodel.frames[int(i)].name + "_frictionCone", friction_cone, 1e1
            )
        if swing_foot_task is not None:
            for task in swing_foot_task:
                frame_translation_residual = crocoddyl.ResidualModelFrameTranslation(
                    self.state, int(task[0]), task[1].translation, nu
                )
                frame_velocity_residual = crocoddyl.ResidualModelFrameVelocity(
                    self.state,
                    int(task[0]),
                    pinocchio.Motion.Zero(),
                    pinocchio.LOCAL_WORLD_ALIGNED,
                    nu,
                )
                foot_track = crocoddyl.CostModelResidual(self.state, frame_translation_residual)
                impulse_foot_vel_cost = crocoddyl.CostModelResidual(
                    self.state, frame_velocity_residual
                )
                cost_model.addCost(
                    self.rmodel.frames[int(task[0])].name + "_footTrack",
                    foot_track,
                    1e7,
                )
                cost_model.addCost(
                    self.rmodel.frames[int(task[0])].name + "_impulseVel",
                    impulse_foot_vel_cost,
                    1e6,
                )

        state_weights = np.array(
            [0.0] * 3 + [500.0] * 3 + [0.01] * (self.rmodel.nv - 6) + [10.0] * self.rmodel.nv
        )
        state_residual = crocoddyl.ResidualModelState(self.state, self.rmodel.defaultState, nu)
        state_activation = crocoddyl.ActivationModelWeightedQuad(state_weights**2)
        state_reg = crocoddyl.CostModelResidual(self.state, state_activation, state_residual)
        if self._fwddyn:
            ctrl_residual = crocoddyl.ResidualModelControl(self.state, nu)
            ctrl_reg = crocoddyl.CostModelResidual(self.state, ctrl_residual)
        else:
            ctrl_residual = crocoddyl.ResidualModelJointEffort(self.state, self.actuation, nu)
            ctrl_reg = crocoddyl.CostModelResidual(self.state, ctrl_residual)
        cost_model.addCost("state_reg", state_reg, 1e1)
        cost_model.addCost("ctrl_reg", ctrl_reg, 1e-3)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        if self._fwddyn:
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state, self.actuation, contact_model, cost_model, 0.0, True
            )
        else:
            dmodel = crocoddyl.DifferentialActionModelContactInvDynamics(
                self.state, self.actuation, contact_model, cost_model
            )
        if self._integrator == "euler":
            model = crocoddyl.IntegratedActionModelEuler(dmodel, 0.0)
        elif self._integrator == "rk4":
            model = crocoddyl.IntegratedActionModelRK(dmodel, crocoddyl.RKType.four, 0.0)
        elif self._integrator == "rk3":
            model = crocoddyl.IntegratedActionModelRK(dmodel, crocoddyl.RKType.three, 0.0)
        elif self._integrator == "rk2":
            model = crocoddyl.IntegratedActionModelRK(dmodel, crocoddyl.RKType.two, 0.0)
        else:
            model = crocoddyl.IntegratedActionModelEuler(dmodel, 0.0)
        return model

    def create_impulse_model(
        self,
        support_foot_ids: List[int],
        swing_foot_task: List[Tuple[int, pinocchio.SE3, Vector3D]],
        JMinvJt_damping: float = 1e-12,
        r_coeff: float = 0.0,
    ) -> crocoddyl.ActionModelImpulseFwdDynamics:
        """Action model for impulse models.

        An impulse model consists of describing the impulse dynamics against a set of
        contacts.

        Args:
            support_foot_ids (List[int]): Ids of the constrained feet (stance feet)
            swing_foot_task (List[Tuple[int, pinocchio.SE3, Vector3D]]): swinging foot tasks
            JMinvJt_damping (float, optional): Damping factor for cholesky decomposition of JMinvJt.
                Defaults to 1e-12.
            r_coeff (float, optional): Restitution coefficient that describes elastic impacts.
                Defaults to 0.0.

        Returns:
            crocoddyl.ActionModelImpulseFwdDynamics: impulse action model
        """
        # Creating a 3D multi-contact model, and then including the supporting foot
        impulse_model = crocoddyl.ImpulseModelMultiple(self.state)
        for i in support_foot_ids:
            support_contact_model = crocoddyl.ImpulseModel3D(
                self.state, int(i), pinocchio.LOCAL_WORLD_ALIGNED
            )
            impulse_model.addImpulse(
                self.rmodel.frames[int(i)].name + "_impulse", support_contact_model
            )

        # Creating the cost model for a contact phase
        cost_model = crocoddyl.CostModelSum(self.state, 0)
        if swing_foot_task is not None:
            for task in swing_foot_task:
                frame_translation_residual = crocoddyl.ResidualModelFrameTranslation(
                    self.state, int(task[0]), task[1].translation, 0
                )
                foot_track = crocoddyl.CostModelResidual(self.state, frame_translation_residual)
                cost_model.addCost(
                    self.rmodel.frames[int(task[0])].name + "_footTrack",
                    foot_track,
                    1e7,
                )

        state_weights = np.array(
            [1.0] * 6 + [10.0] * (self.rmodel.nv - 6) + [10.0] * self.rmodel.nv
        )
        state_residual = crocoddyl.ResidualModelState(self.state, self.rmodel.defaultState, 0)
        state_activation = crocoddyl.ActivationModelWeightedQuad(state_weights**2)
        state_reg = crocoddyl.CostModelResidual(self.state, state_activation, state_residual)
        cost_model.addCost("state_reg", state_reg, 1e1)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        model = crocoddyl.ActionModelImpulseFwdDynamics(self.state, impulse_model, cost_model)
        model.JMinvJt_damping = JMinvJt_damping
        model.r_coeff = r_coeff
        return model

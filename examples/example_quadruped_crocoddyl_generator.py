#! /usr/bin/env python

"""
This script can generates different types of gaits for any quadruped robot using the Crocoddyl library.

In essence, this script can set up different gait problems for a quadruped robot, solves it using trajectory
optimisation techniques provided by Crocoddyl, and optionally visualizes the resulting motion.


Setting up gait problem: All gaits can be defined using high-level gait parameters such as
step_frequencies, duty_cycles, and phase_offsets.
    step frequencies: refers to how many steps a leg takes per second. It essentially dictates the
        cadence or tempo of a leg's movement during locomotion.
    duty cycles: refers to the percentage of time a leg spends in the stance phase
        (in contact with the ground) within a single, complete gait cycle.
    phase offsets: determines the relative timing of each leg's step cycle within the overall gait.
        Imagine each leg having its own independent clock controlling its swing and stance phases.
        Phase offsets are like adjusting the starting time on these clocks.

The `create_generic_gait_problem` method takes these parameters and generates a Crocoddyl
shooting problem that can be solved using the DDP solver.

The resulting solution is a sequence of states and controls that can be used to simulate
the robot's motion.

The script also includes options to display the motion in Meshcat and plot the convergence
of the DDP solver.

This script provides a simple example of how to use the Crocoddyl library to generate
different types of gaits for quadruped robots.

The code can be easily modified to generate different gaits by changing the gait parameters.
For example, you can change the step frequency, duty cycle, phase offset, relative foot target,
foot lift height, duration, and time step to generate different gaits. Few different types of
gaits are defined using these parameters and are provided below.

You can also use the same `create_generic_gait_problem` interface method to generate gaits for
other types of robots, such as bipeds, hexapods, etc.

For more information about the Crocoddyl library, please visit the following website:
https://github.com/loco-3d/crocoddyl/wiki.
"""

import time

import example_robot_data
import numpy as np
import pinocchio

import crocoddyl
from crocoddyl.utils.quadruped import plotSolution
from robot_gait_generator.crocoddyl import CrocoddylGaitProblemsInterface

WITHDISPLAY = True
WITHPLOT = False

# NOTE: the leg order is a list of valid frame names from the robot's description (urdf)
LEG_ORDER = ["lf_foot", "rf_foot", "lh_foot", "rh_foot"]  # these are the names for the hyq robot

# define gait parameters for a few different types of gaits that can be used in this example.
GAIT_PARAMETERS = {
    # NOTE: all parameters are defined following the order in `LEG_ORDER`
    "crawl": {  # => static walk
        "step_frequencies": [1.6] * 4,
        "duty_cycles": [0.8] * 4,
        "phase_offsets": [0.0, 0.25, 0.5, 0.75],
        # relative feet targets refer to the XYZ displacement for each foot per step
        "relative_feet_targets": [[0.3, 0.0, 0.0]] * 4,
        "foot_lift_height": [0.25] * 4,
    },
    "trot": {
        "step_frequencies": [1.6] * 4,
        "duty_cycles": [0.6] * 4,
        "phase_offsets": [0.0, 0.5, 0.5, 0.0],
        "relative_feet_targets": [[0.3, 0.0, 0.0]] * 4,
        "foot_lift_height": [0.25] * 4,
    },
    "flying trot": {
        "step_frequencies": [2.2] * 4,
        "duty_cycles": [0.35] * 4,
        "phase_offsets": [0.0, 0.5, 0.5, 0.0],
        "relative_feet_targets": [[0.5, 0.0, 0.0]] * 4,
        "foot_lift_height": [0.25] * 4,
    },
    "bound": {
        "step_frequencies": [2.0] * 4,
        "duty_cycles": [0.5] * 4,
        "phase_offsets": [0.0, 0.25, 0.5, 0.75],
        "relative_feet_targets": [[0.3, 0.0, 0.0]] * 4,
        "foot_lift_height": [0.25] * 4,
    },
    "pace": {
        "step_frequencies": [2.0] * 4,
        "duty_cycles": [0.5] * 4,
        "phase_offsets": [0.0, 0.5, 0.0, 0.5],
        "relative_feet_targets": [[0.4, 0.0, 0.0]] * 4,
        "foot_lift_height": [0.2] * 4,
    },
    "leap": {
        "step_frequencies": [2.0] * 4,
        "duty_cycles": [0.5] * 4,
        "phase_offsets": [0.0, 0.0, 0.0, 0.0],
        "relative_feet_targets": [[0.3, 0.0, 0.0]] * 4,
        "foot_lift_height": [0.12] * 4,
    },
    "walk": {  # for amble, increase step frequency
        "step_frequencies": [1.5] * 4,
        "duty_cycles": [0.6] * 4,
        "phase_offsets": [0.0, 0.5, 0.1, 0.6],
        "relative_feet_targets": [[0.3, 0.0, 0.0]] * 4,
        "foot_lift_height": [0.05] * 4,
    },
    "gallop": {
        "step_frequencies": [2.4] * 4,
        "duty_cycles": [0.35] * 4,
        "phase_offsets": [0.1, 0.0, 0.6, 0.5],
        "relative_feet_targets": [[0.7, 0.0, 0.0]] * 4,
        "foot_lift_height": [0.2] * 4,
    },
}

# choose the gait to generate for this robot
GAIT_CHOICE = "walk"

if __name__ == "__main__":

    # Loading the robot model
    robot = example_robot_data.load("hyq")

    # Defining the initial state of the robot
    # NOTE: q0 is vital for the optimisation problem!!
    q0 = robot.model.referenceConfigurations["standing"].copy()
    v0 = pinocchio.utils.zero(robot.model.nv)
    x0 = np.concatenate([q0, v0])

    # Setting up the 3d walking problem
    # NOTE: only the pinocchio model is required for creating the gait problem interface
    # and solving. The full `robot` object created above is only for being
    # able to use the visualisation tool in Meshcat
    # For any URDF, create a pinocchio model using:
    # ```
    # model = pinocchio.buildModelFromUrdf(urdf_filename, pinocchio.JointModelFreeFlyer())
    # ```
    gait_problem_interface = CrocoddylGaitProblemsInterface(
        pinocchio_robot_model=robot.model,  # pinocchio model
        default_standing_configuration=q0,
        ee_names=LEG_ORDER,
    )

    # Creating the DDP solver for the locomotion problem
    solver = crocoddyl.SolverFDDP(
        # here we can define the type of gait using high-level gait parameters
        gait_problem_interface.create_generic_gait_problem(
            x0=x0,
            **GAIT_PARAMETERS[GAIT_CHOICE],
            starting_feet_heights=[0.0, 0.0, 0.0, 0.0] * 4,
            duration=10,
            time_step=0.01,
        )
    )

    # Add optional callback functions for verbosity/debugging
    print("*** SOLVE " + "***")
    if WITHPLOT:
        solver.setCallbacks(
            [
                crocoddyl.CallbackVerbose(),
                crocoddyl.CallbackLogger(),
            ]
        )
    else:
        solver.setCallbacks([crocoddyl.CallbackVerbose()])

    # Solving the problem with the DDP solver
    xs = [x0] * (solver.problem.T + 1)
    us = solver.problem.quasiStatic([x0] * solver.problem.T)

    # try to solve the problem in 100 iterations. If the program runs for these many iterations,
    # it did not converge. Generally, for standard gaits, the solver should converge in fewer than
    # 100 iterations. Depending on complexity of motion and robot embodiment, it may take more
    # iterations. E.g. "gallop" gait for hyq robot requires almost 140 iterations.
    num_iter = 100 if GAIT_CHOICE in ["gallop", "leap"] else 200
    solver.solve(xs, us, num_iter, False)
    print("solving done")

    # Display the entire motion
    if WITHDISPLAY:
        display = crocoddyl.MeshcatDisplay(robot, frameNames=LEG_ORDER)
        display.rate = -1
        display.freq = 1
        while True:
            try:
                display.displayFromSolver(solver)
                time.sleep(1.0)
            except KeyboardInterrupt:
                break

    # Plotting the entire motion
    if WITHPLOT:
        plotSolution(solver, figIndex=1, show=False)

        title = "Gait"
        log = solver.getCallbacks()[1]
        crocoddyl.plotConvergence(
            log.costs,
            log.pregs,
            log.dregs,
            log.grads,
            log.stops,
            log.steps,
            figTitle=title,
            figIndex=3,
            show=True,
        )

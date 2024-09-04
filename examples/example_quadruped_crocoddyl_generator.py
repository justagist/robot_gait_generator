"""
This script can generates different types of gaits for any quadruped robot using the Crocoddyl library.

In essence, this script can set up different gait problems for a quadruped robot, solves it using trajectory
optimisation techniques provided by Crocoddyl, and optionally visualizes the resulting motion.
"""

import time

import example_robot_data
import numpy as np
import pinocchio

import crocoddyl
from crocoddyl.utils.quadruped import plotSolution
from robot_gait_generator.crocoddyl import CrocoddylGaitProblemsInterface

WITHDISPLAY = True
WITHPLOT = True

# Loading the robot model
robot = example_robot_data.load("hyq")

# Defining the initial state of the robot
q0 = robot.model.referenceConfigurations["standing"].copy()
v0 = pinocchio.utils.zero(robot.model.nv)
x0 = np.concatenate([q0, v0])

# Setting up the 3d walking problem
ee_names = ["lf_foot", "rf_foot", "lh_foot", "rh_foot"]

gait_problem_interface = CrocoddylGaitProblemsInterface(
    pinocchio_robot_model=robot.model,
    default_standing_configuration=q0,
    ee_names=ee_names,
)


# Setting up gait problem
solver = crocoddyl.SolverFDDP(
    gait_problem_interface.create_generic_gait_problem(
        x0=x0,
        step_frequencies=[1.6] * 4,
        duty_cycles=[0.6] * 4,
        phase_offsets=[0.5, 0.5, 0.0, 0.0],
        relative_feet_targets=[[0.3, 0.0, 0.0]] * 4,
        starting_feet_heights=[0.0, 0.0, 0.0, 0.0] * 4,
        foot_lift_height=0.25,
        duration=8,
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

# try 100 iterations, if the program runs for these many iterations,
# it did not converge.
solver.solve(xs, us, 100, False)
print("solving done")

# Display the entire motion
if WITHDISPLAY:
    display = crocoddyl.MeshcatDisplay(robot, frameNames=ee_names)
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

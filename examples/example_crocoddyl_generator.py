import os
import signal
import sys
import time

import example_robot_data
import numpy as np
import pinocchio

import crocoddyl
from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem, plotSolution
from robot_gait_generator.crocoddyl_gait_problems_interface import CrocoddylGaitProblemsInterface

WITHDISPLAY = True
WITHPLOT = True
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Loading the anymal model
anymal = example_robot_data.load("hyq")

# Defining the initial state of the robot
q0 = anymal.model.referenceConfigurations["standing"].copy()
v0 = pinocchio.utils.zero(anymal.model.nv)
x0 = np.concatenate([q0, v0])

# Setting up the 3d walking problem
ee_names = lfFoot, rfFoot, lhFoot, rhFoot = "lf_foot", "rf_foot", "lh_foot", "rh_foot"
# gait = SimpleQuadrupedalGaitProblem(anymal.model, lfFoot, rfFoot, lhFoot, rhFoot)
gait_problem_interface = CrocoddylGaitProblemsInterface(
    pinocchio_robot_model=anymal.model,
    default_standing_configuration=q0,
    ee_names=ee_names,
)

# Setting up all tasks

solver = crocoddyl.SolverFDDP(
    gait_problem_interface.create_generic_gait_problem(
        x0=x0,
        step_frequencies=[1.6] * 4,
        duty_cycles=[0.6] * 4,
        phase_offsets=[0.5, 0.5, 0.0, 0.0],
        relative_feet_targets=[[0.2, 0.0, 0.1]] * 4,
        starting_feet_heights=[0.2, 0.2, 0.0, 0.0] * 4,
        foot_lift_height=0.25,
        duration=8,
        time_step=0.01,
    )
)

# Added the callback functions
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
solver.solve(xs, us, 100, False)
print("solving done")

# Defining the final state as initial one for the next phase
x0 = solver.xs[-1]

# Display the entire motion
if WITHDISPLAY:
    try:
        import gepetto

        gepetto.corbaserver.Client()
        cameraTF = [2.0, 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]
        display = crocoddyl.GepettoDisplay(
            anymal, 4, 4, cameraTF, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot]
        )
    except Exception:
        display = crocoddyl.MeshcatDisplay(anymal, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
    display.rate = -1
    display.freq = 1
    while True:
        display.displayFromSolver(solver)
        time.sleep(1.0)

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

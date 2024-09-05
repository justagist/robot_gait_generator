# Robot Gait Generator

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)

This package provides components for generating motion trajectories and control actions for different gaits for legged robots. Primarily, this package provides an easy interface using [Crocoddyl](https://github.com/loco-3d/crocoddyl/wiki) to define and generate different types of gaits for legged robots using high-level gait parameters such as step frequency, duty cycle, phase offset, and foot lift height.

**Note:** This library focuses on generating a sequence of optimized control actions for a given gait using trajectory optimization. It does not provide instantaneous control actions that can be used to control a robot. However, the utility tools for feet trajectory generation and gait scheduling can be used independently for control if required.

See [example file](examples/example_quadruped_crocoddyl_generator.py) for usage.

## Features

- Provides interface to easily define gait problem for any legged robot, only requires a URDF.
  - The defined gait problem can be solved using Crocoddyl solvers (see example file). Solution provides dynamically feasible motions in terms of
    joint positions, velocities, torques, etc.
- Allows defining any gait for a legged robot using a few high-level parameters such as step frequency, duty cycle, phase offset, and foot lift height.
- Example parameters for several gait patterns already provided: like walk, trot, pace, gallop, bound, etc.
- Additional utility classes for contact scheduling, generating smooth foot trajectories, etc.

## Installation

### Using [Pixi](https://pixi.sh/latest/) (recommended)

If you have [pixi](https://pixi.sh/latest/) installed, install this package using `pixi install`. Run example using `pixi run python3 examples/example_quadruped_crocoddyl_generator.py`.

### Using pip

Alternative, run `pip install .` (ideally in a virtual environment). And run example using `python3 examples/example_quadruped_crocoddyl_generator.py`.

## Examples

These are a few different types of motions that can be generated using this formulation. All motions are generated from the same code by only changing the gait parameters. All of these (and more) can be reproduced using the [provided example code](examples/example_quadruped_crocoddyl_generator.py).

### Walk

[![walk_gait](https://media.githubusercontent.com/media/justagist/_assets/main/robot_gait_generator/walk.gif)](https://media.githubusercontent.com/media/justagist/_assets/main/robot_gait_generator/walk.gif)

### Flying Trot

[![flying_trot_gait](https://media.githubusercontent.com/media/justagist/_assets/main/robot_gait_generator/flying_trot.gif)](https://media.githubusercontent.com/media/justagist/_assets/main/robot_gait_generator/flying_trot.gif)

### Gallop

[![gallop_gait](https://media.githubusercontent.com/media/justagist/_assets/main/robot_gait_generator/gallop.gif)](https://media.githubusercontent.com/media/justagist/_assets/main/robot_gait_generator/gallop.gif)

### Leap

[![leap_gait](https://media.githubusercontent.com/media/justagist/_assets/main/robot_gait_generator/leap.gif)](https://media.githubusercontent.com/media/justagist/_assets/main/robot_gait_generator/leap.gif)

**NOTE:** Although the examples above use the HyQ robot, any robot with a valid URDF is supported.

## Setting up gait problem

All gaits can be defined using high-level gait parameters such as step_frequencies, duty_cycles, and phase_offsets.

- step frequencies: refers to how many steps a leg takes per second. It essentially dictates the
      cadence or tempo of a leg's movement during locomotion.
- duty cycles: refers to the percentage of time a leg spends in the stance phase
      (in contact with the ground) within a single, complete gait cycle.
- phase offsets: determines the relative timing of each leg's step cycle within the overall gait.
      Imagine each leg having its own independent clock controlling its swing and stance phases.
      Phase offsets are like adjusting the starting time on these clocks.

The `create_generic_gait_problem` interface takes these parameters and generates a Crocoddyl
shooting problem that can be solved using the solvers in Crocoddyl (e.g. DDP).
The resulting solution is a sequence of states and controls that can be used to simulate
the robot's motion.

This package provides a [simple example](examples/example_quadruped_crocoddyl_generator.py) of how to generate
different types of gaits for quadruped robots. The code can be easily modified to generate different gaits by changing the gait parameters. Few different gaits are already defined using these gait parameters in the example file.

You can also use the same `create_generic_gait_problem` interface method to generate gaits for
other types of robots, such as bipeds, hexapods, etc.

For more information about the Crocoddyl library, please visit the following website:
<https://github.com/loco-3d/crocoddyl/wiki>.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions or improvements.

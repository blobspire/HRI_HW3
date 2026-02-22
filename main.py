import pybullet as p
import pybullet_data
import numpy as np
import os
import time
import json
from robot import Panda
from teleop import KeyboardController
from objects import objects


# function that returns the goal positions and rotation around z for each object
# from our lecture notes, these are theta_box, theta_banana, etc.
def get_object_goals():
    box_state = box.get_state()
    banana_state = banana.get_state()
    bottle_state = bottle.get_state()
    goals = {}
    # The arrays added on set up for correct grasp. Position is center so addition moves to graspable edge at top
    goals["box_position"] = box_state["position"] + np.array([0, 0, 0.05])
    goals["box_rotz"] = box_state["euler"][2] + np.pi/2
    goals["banana_position"] = banana_state["position"] + np.array([0, 0, -0.01])
    goals["banana_rotz"] = banana_state["euler"][2] + np.pi/2
    goals["bottle_position"] = bottle_state["position"] + np.array([-0.01, 0, 0.05])
    goals["bottle_rotz"] = bottle_state["euler"][2] + 0.0
    return goals

# function that outputs the actions to reach potential target
def get_object_actions(robot_position, robot_euler, goals):
    actions = {}
    actions["box"] = action_to_goal(robot_position, robot_euler, goals["box_position"], goals["box_rotz"])
    actions["banana"] = action_to_goal(robot_position, robot_euler, goals["banana_position"], goals["banana_rotz"])
    actions["bottle"] = action_to_goal(robot_position, robot_euler, goals["bottle_position"], goals["bottle_rotz"])
    return actions

# function that outputs the next target position and target quaternion if we are 
# reaching for the goal_position and goal_rotz
def action_to_goal(robot_position, robot_euler, goal_position, goal_rotz):
    position_error = goal_position - robot_position
    rotz_error = goal_rotz - robot_euler[2]
    if np.linalg.norm(position_error) > 0.01:
        position_error = position_error / np.linalg.norm(position_error)
    if np.abs(rotz_error) > 0.01:
        rotz_error = rotz_error / np.abs(rotz_error)
    # the gains 0.001 and 0.005 match the default pos_step and rot_step in teleop
    target_position = robot_position + 0.001 * position_error
    target_euler = np.array([np.pi, 0, robot_euler[2] + 0.005 * rotz_error])
    return target_position, np.array(p.getQuaternionFromEuler(target_euler))

# parameters
control_dt = 1. / 240.

# create simulation and place camera
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
# disable keyboard shortcuts so they do not interfere with keyboard control
p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=1.0, 
                                cameraYaw=40.0,
                                cameraPitch=-30.0, 
                                cameraTargetPosition=[0.5, 0.0, 0.2])

# load the objects
urdfRootPath = pybullet_data.getDataPath()
plane = objects.PyBulletObject("plane.urdf", basePosition=[0, 0, -0.625])
table = objects.PyBulletObject("table/table.urdf", basePosition=[0.5, 0, -0.625])
box = objects.YCBObject("003_cracker_box.urdf", basePosition=[0.6, -0.2, 0.09], 
                                                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))
banana = objects.YCBObject("011_banana.urdf", basePosition=[0.7, 0.2, 0.025], 
                                                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))
bottle = objects.YCBObject("006_mustard_bottle.urdf", basePosition=[0.5, 0.05, 0.06], 
                                                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))

# load the robot
jointStartPositions = [0.0, 0.0, 0.0, -2*np.pi/4, 0.0, np.pi/2, np.pi/4, 0.0, 0.0, 0.04, 0.04]
panda = Panda(basePosition=[0, 0, 0],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                jointStartPositions=jointStartPositions)

# teleoperation interface
teleop = KeyboardController()

# run simulation
# you can teleoperate the robot using the keyboard;
# see "teleop.py" for the mapping between keys and motions
state = panda.get_state()
target_position = state["ee-position"]
target_quaternion = state['ee-quaternion']
start_position = np.array(state["ee-position"]) # Initial position
human_interaction_delay = 3.0 # seconds after which we consider the human to have stopped interacting
time_till_robot_control = human_interaction_delay
time_till_probs_print = 0.5 # seconds until we print the next probs
robot_enabled = True
while True:
    # update the target pose
    action = teleop.get_action()
    # If human has interacted, reset time till robot control
    if np.linalg.norm(action) > 0:
        time_till_robot_control = human_interaction_delay
    human_position = target_position + action[0:3]
    human_quaternion = p.multiplyTransforms([0, 0, 0], p.getQuaternionFromEuler(action[3:6]),
                                                [0, 0, 0], target_quaternion)[1]
    human_quaternion = np.array(human_quaternion)

    # share autonomy between human and robot

    # Predict the human goal
    # distance from start to goal / (distance from start to current + distance from current to goal)
    # Get current state
    state = panda.get_state()
    # Get current position
    current_position = np.array(panda.get_state()["ee-position"])
    # Get object goals (these are theta1, theta2, theta3)
    goals = get_object_goals()
    object_positions = np.array([goals["box_position"], goals["banana_position"], goals["bottle_position"]])
    start_to_goal = np.linalg.norm(start_position - object_positions, axis=1)
    start_to_current = np.linalg.norm(start_position - current_position)
    current_to_goal = np.linalg.norm(current_position - object_positions, axis=1)
    BETA = 12
    probs = np.exp(BETA * start_to_goal) / (np.exp(BETA * start_to_current + BETA * current_to_goal)) # Purpose of beta: human isn't perfect so beta allows us to make things more or less deterministic.
    # Normalize to convert to probabilities
    probs = probs / np.sum(probs) # This is the theta probs that human wants to grab each object
    # Only print probs every 0.5 seconds
    time_till_probs_print -= control_dt
    if time_till_probs_print <= 0:
        time_till_probs_print = 0.5
        print(f"Probs: {probs}")

    # target_position like a from notes, human_position like a_h
    # We have 'human_position' using teleop and 'get_object_actions' which returns 'action' (an array of actions for each object)
    # Grab the robot action for the most likely object and then add to human_position
    # Base alpha on whether human is interacting or not
    if time_till_robot_control <= 0:
        ALPHA = 1.0 # If no interaction, fully robot control
    elif robot_enabled:
        ALPHA = 0.3 # Tunable blending factor between 0 and 1 (0: fully human, 1: fully robot)
    elif not robot_enabled:
        ALPHA = 0.0 # If robot control disabled, fully human control
    CONFIDENCE_THRESHOLD = 0.6
    object_actions = get_object_actions(current_position, state["ee-euler"], goals)
    max_index = np.argmax(probs)
    most_likely_object_index = max_index if probs[max_index] > CONFIDENCE_THRESHOLD else -1
    if (most_likely_object_index > -1):
        robot_action = object_actions[list(object_actions.keys())[most_likely_object_index]]
        blended_position = (1 - ALPHA) * human_position + ALPHA * robot_action[0]
        blended_quaternion = (1 - ALPHA) * human_quaternion + ALPHA * robot_action[1]
    else:
        blended_position = human_position
        blended_quaternion = human_quaternion
    # get_object_actions returns tuple of position and quaternion
    # Decrement time till robot control by sim step size
    time_till_robot_control -= control_dt

    target_position = blended_position
    target_quaternion = blended_quaternion

    # impose workspace limit
    if target_position[2] < 0.02:
        target_position[2] = 0.02

    # move to the target pose
    panda.move_to_pose(ee_position=target_position, ee_quaternion=target_quaternion, positionGain=1)

    # open or close the gripper
    if action[6] == +1:
        panda.open_gripper()
    elif action[6] == -1:
        panda.close_gripper()

    # print when "." is pressed
    if action[7] == +1:
        print("button pressed")

    # Disable/enable robot control if "," if pressed
    if action[7] == -1:
        robot_enabled = not robot_enabled
        print(f"Robot control {'enabled' if robot_enabled else 'disabled'}")
        time.sleep(0.25)

    # step simulation
    p.stepSimulation()
    time.sleep(control_dt)
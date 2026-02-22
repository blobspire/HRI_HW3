import pybullet as p
import numpy as np


# class to collect keyboard inputs and convert them to robot actions
# here we are using the keyboard to control the top-down robot arm
class KeyboardController:

    # pos_step and rot_step define the magnitude of the position and rotation inputs
    # increasing this will mean pressing a key applies a larger motion
    def __init__(self, pos_step=0.001, rot_step=0.005):
        self.pos_step = pos_step
        self.rot_step = rot_step

        # mapping from keys to robot motions
        self.key_map = {ord("w"): np.array([+1, 0, 0, 0, 0, 0, 0, 0]),  # +x
                        ord("s"): np.array([-1, 0, 0, 0, 0, 0, 0, 0]),  # -x
                        ord("a"): np.array([0, +1, 0, 0, 0, 0, 0, 0]),  # +y
                        ord("d"): np.array([0, -1, 0, 0, 0, 0, 0, 0]),  # -y
                        ord("q"): np.array([0, 0, +1, 0, 0, 0, 0, 0]),  # +z
                        ord("e"): np.array([0, 0, -1, 0, 0, 0, 0, 0]),  # -z
                        ord("p"): np.array([0, 0, 0, 0, 0, +1, 0, 0]),  # yaw +
                        ord("l"): np.array([0, 0, 0, 0, 0, -1, 0, 0]),  # yaw -
                        ord("z"): np.array([0, 0, 0, 0, 0, 0, +1, 0]),  # open gripper
                        ord("x"): np.array([0, 0, 0, 0, 0, 0, -1, 0]),  # close gripper
                        ord("."): np.array([0, 0, 0, 0, 0, 0, 0, +1]),  # toggle
                        ord(","): np.array([0, 0, 0, 0, 0, 0, 0, -1]),  # disable robot control
                        }
    
    # read the keyboard inputs and convert them to robot actions
    # returns an array: [x, y, z, roll, pitch, yaw, gripper, toggle]
    def get_action(self):
        keys = p.getKeyboardEvents()
        action = np.zeros(8, dtype=np.float32)

        for k, v in self.key_map.items():
            if k in keys and (keys[k] & p.KEY_IS_DOWN):
                action += v

        action[0:3] *= self.pos_step
        action[3:6] *= self.rot_step
        return action
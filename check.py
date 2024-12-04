# import cv2
from dm_control import mujoco
import cv2
import numpy as np
import random
import ikpy.chain
import ikpy.utils.plot as plot_utils
import transformations as tf
import PIL
import sys

# Load dm_control model
model = mujoco.Physics.from_xml_path('assets/banana.xml')
model_bgr = mujoco.Physics.from_xml_path('assets/banana_bgr.xml')

# Load the robot arm chain from the URDF file
my_chain = ikpy.chain.Chain.from_urdf_file("assets/a1_right.urdf")

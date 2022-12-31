#!/usr/bin/env python

##################################################################################
# MIT License                                                                    #
#                                                                                #
# Copyright (c) 2022 James Mount                                                 #
#                                                                                #
# Permission is hereby granted, free of charge, to any person obtaining a copy   #
# of this software and associated documentation files (the "Software"), to deal  #
# in the Software without restriction, including without limitation the rights   #
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      #
# copies of the Software, and to permit persons to whom the Software is          #
# furnished to do so, subject to the following conditions:                       #
#                                                                                #
# The above copyright notice and this permission notice shall be included in all #
# copies or substantial portions of the Software.                                #
#                                                                                #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  #
# SOFTWARE.                                                                      #
##################################################################################

###############
### MODULES ###
###############

import pytest

import os
import pathlib
import numpy as np

import cv2
import open3d as o3d
import spatialmath as sm
from PIL import Image, ImageChops

from robotools.roboframes import *

SCRIPT_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))


###################################
### ROBOFRAME BASE CLASS TESTS ###
###################################

# Constructor Tests
def test_roboframe_class_constructor():
    frame = RoboFrame(1)
    assert frame.frame_id == 1

# Test add_data and has field
def test_roboframe_class_add_data():
    frame = RoboFrame(1)
    frame.add_data('test_0', 1)

    assert frame.test_0 == 1
    assert frame.has_field('test_0') == True
    assert frame.has_field('test_1') == False

    frame.add_data(['test_2', 'test_3'], ['string', 10.1])
    assert frame.test_2 == 'string'
    assert frame.test_3 == 10.1

    frame.add_data(['test_4'], 'string')
    frame.add_data('test_5', [10])
    assert frame.test_4 == 'string'
    assert frame.test_5 == 10

    with pytest.raises(ValueError):
        frame.add_data(['test_6', 'test_7'], [10])
        frame.add_data('test_8', ['string', 10])


# Test set_pose and get_pose_data
def test_roboframe_class_timestamp():
    frame = RoboFrame(1)

    assert frame.get_pose_data() == None
    frame.pose = 10.1
    assert frame.get_pose_data() == None

    frame.pose = sm.SE3()
    data = frame.get_pose_data()
    assert data['pos_x'] == 0
    assert data['pos_y'] == 0
    assert data['pos_z'] == 0
    assert data['quat_w'] == 1
    assert data['quat_x'] == 0
    assert data['quat_y'] == 0
    assert data['quat_z'] == 0

    assert frame.set_pose(pos_x=1) == False
    assert frame.set_pose(pos_x=1, pos_y=2, pos_z=3, quat_w=0, quat_x=1, quat_y=0, quat_z=0) == True
    
    data = frame.get_pose_data()
    assert data['pos_x'] == 1
    assert data['pos_y'] == 2
    assert data['pos_z'] == 3
    assert data['quat_w'] == 0
    assert data['quat_x'] == 1
    assert data['quat_y'] == 0
    assert data['quat_z'] == 0

    assert frame.set_pose(pos_x=1, pos_y=2, pos_z=3) == True
    data = frame.get_pose_data()
    assert data['pos_x'] == 1
    assert data['pos_y'] == 2
    assert data['pos_z'] == 3
    assert data['quat_w'] == 1
    assert data['quat_x'] == 0
    assert data['quat_y'] == 0
    assert data['quat_z'] == 0

    assert frame.set_pose(quat_w=0, quat_x=0, quat_y=1, quat_z=0) == True
    data = frame.get_pose_data()
    assert data['pos_x'] == 0
    assert data['pos_y'] == 0
    assert data['pos_z'] == 0
    assert data['quat_w'] == 0
    assert data['quat_x'] == 0
    assert data['quat_y'] == 1
    assert data['quat_z'] == 0


###################################
### ROBOFRAME FILE CLASS TESTS ###
###################################

# Using RoboFrameImage instead due to abstract base class
# Test error is raised
def test_constructor_error():
    with pytest.raises(TypeError):
        RoboFrameFile(1, "/path/to/file/001.png")

### TESTING CONSTRUCTORS ###
# Testing that frame id and filepath arguments to constructor are correctly passed
def test_constructor_simple():
    frame = RoboFrameImage(0, "/path/to/file/001.png")

    assert frame.frame_id == 0
    assert frame.filepath == pathlib.Path("/path/to/file/001.png")

# TESTING FILENAME PROPERTIES
# Testing filepath with the form <id-number>.<extension> returns correct properties
def test_file_properties_id_with_extension():
    frame = RoboFrameImage(1, "/path/to/file/001.png")

    assert frame.filestem == "001"
    assert frame.filename == "001.png"
    assert frame.rootpath == pathlib.Path("/path/to/file")
    assert frame.extension == "png"
    assert frame.prefix == None
    assert frame.user_notes == None


# Testing filepath with the form <id-number> returns correct properties
def test_file_properties_id_without_extension():
    frame = RoboFrameImage(1, "/path/to/file/001")

    assert frame.filestem == "001"
    assert frame.filename == "001"
    assert frame.rootpath == pathlib.Path("/path/to/file")
    assert frame.extension == None
    assert frame.prefix == None
    assert frame.user_notes == None


# Testing filepath with the form <prefix>_<id-number>.<extension> returns correct properties
def test_file_properties_prefix_id_with_extension():
    frame = RoboFrameImage(1, "/path/to/file/frame_001.png")

    assert frame.filestem == "frame_001"
    assert frame.filename == "frame_001.png"
    assert frame.rootpath == pathlib.Path("/path/to/file")
    assert frame.extension == "png"
    assert frame.prefix == "frame"
    assert frame.user_notes == None


# Testing filepath with the form <prefix>_<id-number> returns correct properties
def test_file_properties_prefix_id_without_extension():
    frame = RoboFrameImage(1, "/path/to/file/frame_001")

    assert frame.filestem == "frame_001"
    assert frame.filename == "frame_001"
    assert frame.rootpath == pathlib.Path("/path/to/file")
    assert frame.extension == None
    assert frame.prefix == "frame"
    assert frame.user_notes == None


# Testing filepath with the form <prefix>_<user-notes>_<id-number>.<extension> returns correct properties
def test_file_properties_prefix_usernotes_id_with_extension_simple():
    frame = RoboFrameImage(1, "/path/to/file/frame_user-notes_001.png")

    assert frame.filestem == "frame_user-notes_001"
    assert frame.filename == "frame_user-notes_001.png"
    assert frame.rootpath == pathlib.Path("/path/to/file")
    assert frame.extension == "png"
    assert frame.prefix == "frame"
    assert frame.user_notes == "user-notes"


# Testing filepath with the form <prefix>_<user-notes>_<id-number> returns correct properties
def test_file_properties_prefix_usernotes_id_without_extension_simple():
    frame = RoboFrameImage(1, "/path/to/file/frame_user-notes_001")

    assert frame.filestem == "frame_user-notes_001"
    assert frame.filename == "frame_user-notes_001"
    assert frame.rootpath == pathlib.Path("/path/to/file")
    assert frame.extension == None
    assert frame.prefix == "frame"
    assert frame.user_notes == "user-notes"


# Testing filepath with the form <prefix>_<user-notes>_<id-number>.<extension> returns correct properties
# when user notes includes underscores
def test_file_properties_prefix_usernotes_id_with_extension_complex():
    frame = RoboFrameImage(1, "/path/to/file/frame_user_notes_001.png")

    assert frame.filestem == "frame_user_notes_001"
    assert frame.filename == "frame_user_notes_001.png"
    assert frame.rootpath == pathlib.Path("/path/to/file")
    assert frame.extension == "png"
    assert frame.prefix == "frame"
    assert frame.user_notes == "user_notes"


# Testing filepath with the form <prefix>_<user-notes>_<id-number> returns correct properties
def test_file_properties_prefix_usernotes_id_without_extension_complex():
    frame = RoboFrameImage(1, "/path/to/file/frame_user_notes_001")

    assert frame.filestem == "frame_user_notes_001"
    assert frame.filename == "frame_user_notes_001"
    assert frame.rootpath == pathlib.Path("/path/to/file")
    assert frame.extension == None
    assert frame.prefix == "frame"
    assert frame.user_notes == "user_notes"


###################################
### ROBOFRAME IMAGE CLASS TESTS ###
###################################

def test_roboframe_image_read():
    filepath = SCRIPT_DIR / "data/starry_night.jpg"
    frame = RoboFrameImage(1, filepath)

    for image_format in [ImageFormat.OPENCV, ImageFormat.PIL]:
            img1 = frame.read(image_format=image_format)
            if image_format == ImageFormat.OPENCV:
                img2 = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
                assert (img1.shape == img2.shape and not(np.bitwise_xor(img1,img2).any())) == True
            else:
                img2 = Image.open(str(filepath))
                assert (ImageChops.difference(img1, img2)).getbbox() == None


def test_roboframe_image_read_grayscale():
    filepath = SCRIPT_DIR / "data/starry_night.jpg"
    frame = RoboFrameImage(1, filepath)

    for image_format in [ImageFormat.OPENCV, ImageFormat.PIL]:
            img1 = frame.read(image_format=image_format, colour=False)
            if image_format == ImageFormat.OPENCV:
                img2 = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
                assert (img1.shape == img2.shape and not(np.bitwise_xor(img1,img2).any())) == True
            else:
                img2 = Image.open(str(filepath)).convert("L")
                assert (ImageChops.difference(img1, img2)).getbbox() == None



###################################
### ROBOFRAME IMAGE CLASS TESTS ###
###################################

def test_roboframe_image_read():
    filepath = SCRIPT_DIR / "data/fragment.ply"

    frame = RoboFramePointCloud(1, filepath)
    pcd1 = frame.read()

    pcd2 = o3d.io.read_point_cloud(str(filepath))

    pcd1_pts = np.asarray(pcd1.points)
    pcd1_col = np.asarray(pcd1.colors)
    pcd2_pts = np.asarray(pcd2.points)
    pcd2_col = np.asarray(pcd2.colors)
    assert np.sum(pcd1_pts - pcd2_pts) == 0
    assert np.sum(pcd1_col - pcd2_col) == 0
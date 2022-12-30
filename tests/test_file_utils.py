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
from PIL import Image, ImageChops

from robotools.defines import FrameType
from robotools.file_utils import *

SCRIPT_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))

################################
### FILE UTIL FUNCTION TESTS ###
################################

# Testing extension_from_filepath function
# Returns correct extension with an absolute path
def test_extension_from_filepath_absolute():
    filepath = "/path/to/file/001.png"
    assert extension_from_filepath(filepath) == "png"

# Returns correct extension with relative paths
def test_extension_from_filepath_relative():
    filepath = "001.png"
    assert extension_from_filepath(filepath) == "png"

    filepath = "relative/001.png"
    assert extension_from_filepath(filepath) == "png"


# Testing frametype_from_extension function
# Returns correct frame type for image extensions
def test_frametype_from_extension_image_types():
    for _type in supported_image_types():
        assert frametype_from_extension(_type) == FrameType.IMAGE

# Returns correct frame type for point cloud extensions
def test_frametype_from_extension_pointcloud_types():
    for _type in supported_pointcloud_types():
        assert frametype_from_extension(_type) == FrameType.POINTCLOUD

# Returns correct frame type for csv extensions
def test_frametype_from_extension_csv_types():
    for _type in supported_csv_types():
        assert frametype_from_extension(_type) == FrameType.CSVDATA

# Returns correct frame type for unknown extensions
def test_frametype_from_extension_unknown_types():
    assert frametype_from_extension("some-extension") == FrameType.UNKNOWN


# Testing frametype_from_filepath function
# Returns correct frame type for absolute and relative file paths
def test_frametype_from_filepath():
    filepath = "/path/to/file/001.png"
    assert frametype_from_filepath(filepath) == FrameType.IMAGE

    filepath = "001.png"
    assert frametype_from_filepath(filepath) == FrameType.IMAGE

    filepath = "relative/001.png"
    assert frametype_from_filepath(filepath) == FrameType.IMAGE


# Testing read_image using OpenCV format
def test_read_image_opencv_colour():
    filepath = SCRIPT_DIR / "data/starry_night.jpg"
    img1 = read_image(filepath)
    img2 = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
    assert (img1.shape == img2.shape and not(np.bitwise_xor(img1,img2).any())) == True

def test_read_image_opencv_grayscale():
    filepath = SCRIPT_DIR / "data/starry_night.jpg"
    img1 = read_image(filepath, colour=False)
    img2 = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
    assert (img1.shape == img2.shape and not(np.bitwise_xor(img1,img2).any())) == True

def test_read_image_opencv_auto_colour():
    filepath = SCRIPT_DIR / "data/starry_night.jpg"
    img1 = read_image(filepath, colour=None)
    img2 = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
    assert (img1.shape == img2.shape and not(np.bitwise_xor(img1,img2).any())) == True

def test_read_image_opencv_auto_grayscale():
    filepath = SCRIPT_DIR / "data/starry_night_gray.jpg"
    img1 = read_image(filepath, colour=None)
    img2 = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
    assert (img1.shape == img2.shape and not(np.bitwise_xor(img1,img2).any())) == True

# Testing read_image using PIL format
def test_read_image_pil_colour():
    filepath = SCRIPT_DIR / "data/starry_night.jpg"
    img1 = read_image(filepath, ImageFormat.PIL)
    img2 = Image.open(str(filepath))
    assert (ImageChops.difference(img1, img2)).getbbox() == None

def test_read_image_pil_grayscale():
    filepath = SCRIPT_DIR / "data/starry_night.jpg"
    img1 = read_image(filepath, ImageFormat.PIL, colour=False)
    img2 = Image.open(str(filepath)).convert("L")
    assert (ImageChops.difference(img1, img2)).getbbox() == None

def test_read_image_pil_auto_colour():
    filepath = SCRIPT_DIR / "data/starry_night.jpg"
    img1 = read_image(filepath, ImageFormat.PIL, colour=None)
    img2 = Image.open(str(filepath))
    assert (ImageChops.difference(img1, img2)).getbbox() == None

def test_read_image_pil_auto_grayscale():
    filepath = SCRIPT_DIR / "data/starry_night_gray.jpg"
    img1 = read_image(filepath, ImageFormat.PIL, colour=None)
    img2 = Image.open(str(filepath))
    assert (ImageChops.difference(img1, img2)).getbbox() == None

# Test read_image
def test_read_image():
    filepath = SCRIPT_DIR / "data/starry_night.jpg"

    img = read_image(filepath, image_format=ImageFormat.OPENCV)
    assert type(img) == np.ndarray
    
    img = read_image(filepath, image_format=ImageFormat.PIL)
    assert type(img) != np.ndarray

    with pytest.raises(ValueError):
        read_image(filepath, -1)

# Testing read_pointcloud
def test_read_pointcloud():
    filepath = SCRIPT_DIR / "data/fragment.ply"
    pcd1 = read_pointcloud(filepath)
    pcd2 = o3d.io.read_point_cloud(str(filepath))

    pcd1_pts = np.asarray(pcd1.points)
    pcd1_col = np.asarray(pcd1.colors)
    pcd2_pts = np.asarray(pcd2.points)
    pcd2_col = np.asarray(pcd2.colors)
    assert np.sum(pcd1_pts - pcd2_pts) == 0
    assert np.sum(pcd1_col - pcd2_col) == 0
    



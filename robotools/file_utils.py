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

"""The RoboTools File Utils module provides utility functions to manipulate file paths
and read files.
"""

###############
### MODULES ###
###############

import pathlib
from typing import List, Union

import numpy as np

import cv2
from PIL import Image
import open3d as o3d

from robotools.defines import FrameType, ImageFormat


########################
### PUBLIC FUNCTIONS ###
########################

def supported_image_types() -> List:
    """Get the list of supported image extensions. The period is not included.

    Returns:
        List: The list of supported image types.
    """
    return ['bmp', 'pbm', 'pgm', 'ppm', 'jpeg', 'jpg', 'jpe', 'jp2', 'tiff', 'tif', 'png']


def supported_pointcloud_types() -> List:
    """Get the list of supported point cloud extensions. The period is not included.

    Returns:
        List: The list of supported point cloud types.
    """
    return ['ply', 'pcd']


def supported_csv_types() -> List:
    """Get the list of supported CSV file extensions. The period is not included.

    Returns:
        List: The list of supported CSV file types.
    """
    return ['csv']


def extension_from_filepath(filepath: pathlib.Path) -> str:
    """Gets the extension for a file given its filepath. The filepath can be relative or absolute.

    Args:
        filepath (pathlib.Path): The relative or absolute path to the file.

    Returns:
        str: The extension for the file without the period.
    """
    filepath = pathlib.Path(filepath)
    if len((filepath.name).rsplit('.',1)) == 2:
        return (filepath.name).rsplit('.',1)[1]
    return None


def frametype_from_extension(extension: str) -> FrameType:
    """Returns the FrameType given the file extension.

    Args:
        extension (str): The file extension without the period.

    Returns:
        FrameType: The FrameType for the given extension.
    """

    if extension in supported_image_types():
        return FrameType.IMAGE
    elif extension in supported_pointcloud_types():
        return FrameType.POINTCLOUD
    elif extension in supported_csv_types():
        return FrameType.CSVDATA
    
    return FrameType.UNKNOWN


def frametype_from_filepath(filepath: pathlib.Path) -> FrameType:
    """Returns the FrameType given the filepath.

    Args:
        filepath (pathlib.Path): The relative or absolute path to the file.

    Returns:
        FrameType: The FrameType for the given filepath.
    """
    extension = extension_from_filepath(filepath)
    return frametype_from_extension(extension)


def read_image(filepath: pathlib.Path, image_format: ImageFormat = ImageFormat.OPENCV, colour: bool=True) -> Union[np.ndarray, Image.Image]:
    """Reads an image either as an OpenCV image (Numpy array, default behaviour) or as a PIL Image.

    Args:
        filepath (pathlib.Path): The file path to the image to be read.
        image_format (ImageFormat, optional): The format to use. Options are ImageFormat.OPENCV or ImageFormat.PIL. Defaults to ImageFormat.OPENCV.
        colour (bool, optional): Set to False to read the image as grayscale. Defaults to True. 

    Raises:
        ValueError: If the passed image_format is not known.

    Returns:
        Union[np.ndarray, Image.Image]: The image.
    """


    # Check if auto-detect if colour or grayscale is on, or user wants specific format
    if image_format == ImageFormat.OPENCV:
        return read_image_opencv(filepath, colour)
    elif image_format == ImageFormat.PIL:
        return read_image_pil(filepath, colour)

    # Raise value error
    raise ValueError("Unknown image format %d."%(image_format))
    

def read_pointcloud(filepath: pathlib.Path) -> o3d.geometry.PointCloud:
    """Reads a point cloud and returns an Open3d Geometry PointCloud object.

    Args:
        filepath (pathlib.Path): The file path to the point cloud to be read.

    Returns:
        o3d.geometry.PointCloud: The read point cloud.
    """

    pcd = o3d.io.read_point_cloud(str(filepath))
    return pcd

#########################
### PRIVATE FUNCTIONS ###
#########################


def read_image_opencv(filepath: pathlib.Path, colour: bool = True) -> np.ndarray:
    """Reads an image as an OpenCV image (Numpy array).

    Args:
        filepath (pathlib.Path): The file path to the image to be read.
        colour (bool, optional): Set to False to read the image as grayscale. Defaults to True.

    Returns:
        np.ndarray: The read image.
    """

    # Check if auto-detect if colour or grayscale is on, or user wants specific format
    if colour != None:
        if colour:
            img = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
        b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
        if (b==g).all() and (b==r).all():
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Return
    return img


def read_image_pil(filepath: pathlib.Path, colour: bool = True) -> Image.Image:
    """Reads an image into a PIL.Image.Image object. 

    Args:
        filepath (pathlib.Path): The file path to the image to be read.
        colour (bool): Set to False to read the image as grayscale. Defaults to True.

    Returns:
        Image.Image: The read image.
    """
    
    # PIL appears to automatically choose the best "mode" (grayscale, RGB, etc.)
    img = Image.open(str(filepath))

    # Force to grayscale
    if colour == False and img.mode != "L":
        img = img.convert('L')

    # Return
    return img
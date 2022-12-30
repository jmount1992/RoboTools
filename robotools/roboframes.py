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

"""The RoboTools RoboFrames modules defines classes that store individual frame
metadata and functions for file I/O.
"""

###############
### MODULES ###
###############

import pathlib
import numpy as np
from typing import Tuple, Union
from abc import ABC, abstractmethod

import open3d as o3d
from PIL import Image

from robotools.defines import ImageFormat
from robotools.file_utils import extension_from_filepath, read_image, read_pointcloud

###############
### CLASSES ###
###############


### ROBO FRAME BASE ###
class RoboFrameBase():
    """The base class inherited by every RoboTools frame type.

    Attributes:
        frame_id (int): the frame ID number.
    """

    def __init__(self, frame_id: int) -> None:
        """The constructor for the RoboFrameBase class.

        Args:
            frame_id (int): the frame ID number.
        """
        self.frame_id = int(frame_id)


### ROBO FRAME CSV ###
class RoboFrameCSV(RoboFrameBase):
    """The RoboTools class for storing a line of data contained within a CSV file, or similar file type.
    Each line in the CSV should be its own RoboFrameCSV object where the attributes are the CSV headers,
    and the values for each attribute is the value for that header for that particular line.

    The class is derived from the RoboFrameBase class.
    """

    def __init__(self, frame_id: int, fields: Tuple = (), values: Tuple = ()) -> None:
        """The constructor for the RoboFrameCSV class.

        Args:
            frame_id (int): the frame ID number.
            fields (Tuple, optional): the fields (CSV headers) for the object. These fields will become the object attributes. Defaults to ().
            values (Tuple, optional): the values for the corresponding fields. Defaults to ().

        Raises:
            ValueError: if the length for fields and values is not equal.
        """
        super().__init__(frame_id)

        if len(fields) != len(values):
            raise ValueError("The number of fields and values must be equal.")

        for field, val in zip(fields, values):
            self.add_data(field.lower(), val)


    def add_data(self, name: str, value) -> None:
        """Adds a field (object attribute) to the object.

        Args:
            name (str): the name of the field to be added.
            value (_type_): the value for said field/attribute.
        """
        setattr(self, name.lower(), value)

### ROBO FRAME FILE ###
class RoboFrameFile(RoboFrameBase, ABC):
    """An abstract base class to be used when frame data is stored as individual files (e.g., images and point clouds).
    This abstract base class provides properties for common file I/O tasks (e.g., getting the file extension or filename)

    The class is derived from the RoboFrameBase class.

    Attributes:
        filepath (Pathlib.path): the absolute file path.
        filestem (str): the name of the file without the path or extension.
        filename (str): the name of the file, this includes the file extension.
        rootpath (Pathlib.Path): the absolute path to the parent folder of the file.
        extension (str): the file extension without the period (e.g., 'png' not '.png').
        prefix (str): the prefix for the file. Will be set to None if there is no prefix.
        user_notes (str): the user notes contained within the file name. Will be set to None if there are no user notes.
    """

    def __init__(self, frame_id: int, filepath: pathlib.Path) -> None:
        """The class constructor.

        Args:
            frame_id (int): the frame ID number.
            filepath (pathlib.Path): the absolute path to the file.
        """
        super().__init__(frame_id)
        self.filepath = pathlib.Path(filepath)

    # Filename properties
    @property
    def filestem(self) -> str:
        """The name of the file without the path or extension (e.g., '/<path-to-file>/<filestem>.<extension>').

        Returns:
            str: the filestem for the frame
        """
        return self.filepath.stem

    @property
    def filename(self) -> str:
        """the name of the file, this includes the file extension. (e.g., '/<path-to-file>/<filestem>.<extension>').

        Returns:
            str: the filename for the frame
        """
        return self.filepath.name

    @property
    def rootpath(self) -> pathlib.Path:
        """The absolute path to the parent folder for the frame.

        Returns:
            str: the absolute path to the parent folder for the frame.
        """
        return self.filepath.parent

    @property
    def extension(self) -> str:
        """The extension for the frame without the period

        Returns:
            str: The extension for the frame without the period. Will return None if the file has no extension.
        """
        return extension_from_filepath(self.filepath)

    @property
    def prefix(self) -> str:
        """The prefix for the frame.

        Returns:
            str: The prefix for the frame. Will return None if the file has no prefix.
        """
        if len((self.filepath.name).split('_',1)) == 2:
            return (self.filepath.name).split('_',1)[0]
        return None
    
    @property
    def user_notes(self) -> str:
        """The user notes contained within the filename.

        Returns:
            str: The user notes contained within the filename. Will return None if no user notes exists.
        """
        if len((self.filepath.name).split('_')) <= 2:
            return None
        return ((self.filepath.name).split('_',1)[1]).rsplit('_',1)[0]

    @abstractmethod
    def read(self, **kwargs): # pragma: no cover
        """To be implemented in each dervied class.
        """
        pass


### ROBO FRAME IMAGE ###
class RoboFrameImage(RoboFrameFile):
    """The RoboTools class for images. Each image within a set should be its own RoboFrameImage
    object.

    The class is derived from the RoboFrameFile class.
    """

    def __init__(self, frame_id: int, filepath: pathlib.Path) -> None:
        """The class constructor.

        Args:
            frame_id (int): the frame ID number.
            filepath (pathlib.Path): the absolute path to the image file.
        """
        super().__init__(frame_id, filepath)

    
    def read(self, **kwargs) -> Union[np.ndarray, Image.Image]:
        """Reads an image file and returns the data. The image can be either read in as
        an OpenCV image (numpy array) or as a PIL image. The image can be forced to be read
        in as colour, grayscale, or automatically determined.

        Kwargs:
            image_format (ImageFormat): the format for the returned image data. Defaults to ImageFormat.OPENCV
            colour (bool): used to force the image colour type. Colour (True), grayscale (False), or if the
                colour should be automatically determined (None). Defaults to True.

        Returns:
            Union[np.ndarray, Image.Image]: the returned image.
        """
        image_format = kwargs.get('image_format', ImageFormat.OPENCV)
        colour = kwargs.get('colour', True)
        return read_image(self.filepath, image_format, colour)


### ROBO FRAME POINT CLOUD ###
class RoboFramePointCloud(RoboFrameFile):
    """The RoboTools class for point clouds. Each point cloud within a set should be its own
    RoboFramePointCloud object.

    The class is derived from the RoboFrameFile class.
    """

    def __init__(self, frame_id: int, filepath: pathlib.Path) -> None:
        """The class constructor.

        Args:
            frame_id (int): the frame ID number.
            filepath (pathlib.Path): the absolute path to the point cloud file.
        """
        super().__init__(frame_id, filepath)

    
    def read(self, **kwargs) -> o3d.geometry.PointCloud:
        """Reads a point cloud file and returns the data as an Open3D point cloud object.

        Returns:
            o3d.geometry.PointCloud: the returned point cloud.
        """
        return read_pointcloud(self.filepath)

########################
### PUBLIC FUNCTIONS ###
########################



#########################
### PRIVATE FUNCTIONS ###
#########################

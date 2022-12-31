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
from typing import Tuple, Union, Any, Dict
from abc import ABC, abstractmethod

import open3d as o3d
from PIL import Image
import spatialmath as sm

from robotools.defines import ImageFormat, PoseComponents
from robotools.file_utils import extension_from_filepath, read_image, read_pointcloud

###############
### CLASSES ###
###############


### ROBO FRAME ###
class RoboFrame():
    """The RoboFrame class.

    Attributes:
        frame_id (int): the frame ID number.
        timestamp (float): the timestamp for the frame. Defaults to None.
        pose (sm.SE3): the pose for the frame. Defaults to None.
    """

    def __init__(self, frame_id: int, timestamp: float = None, pose: sm.SE3 = None) -> None:
        """The constructor for the RoboFrameBase class.

        Args:
            frame_id (int): the frame ID number.
            timestamp (float): the timestamp for the frame. Defaults to None.
            pose (sm.SE3): the pose for the frame. Defaults to None.
        """
        self.frame_id = int(frame_id)
        self.timestamp = timestamp
        self.pose = pose


    def add_data(self, field: Union[str, Tuple], value: Union[Any, Tuple]) -> None:
        """Adds a field (object attribute) to the object.

        Args:
            field (str): the name of the field(s)/attribute(s) to be added. The field(s) will be converted to be all lowercase.
            value (_type_): the value(s) for said field(s)/attribute(s).

        Raises:
            ValueError: if the length for fields and values is not equal.
        """
        if isinstance(field, str):
            field = tuple([field])
        elif isinstance(field, list):
            field = tuple(field)

        if isinstance(value, str):
            value = tuple([value])
        elif isinstance(value, list):
            value = tuple(value)
        elif isinstance(value, tuple) is False:
            value = tuple([value])
        
        if len(field) != len(value):
            raise ValueError("The number of fields and values must be equal.")

        for fld, val in zip(field, value):
            setattr(self, fld.lower(), val)


    def has_field(self, field: str) -> bool:
        """Checks to see if a field exists.

        Args:
            field (str): the name of the field. The field will be converted to be all lowercase.

        Returns:
            bool: true if the field exists.
        """
        return hasattr(self, field.lower())


    def get_pose_data(self) -> Dict:
        """Gets the pose data associated with the frame as a dictionary. The keys in the dictionary will be
        [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z].

        Returns:
            Dict: the pose data as a dictionary or None if the pose is not set or not a Spatial Maths SE(3) object.
        """
        if self.pose is None:
            return None
        elif isinstance(self.pose, sm.SE3) is False:
            return None

        data = {}
        data['pos_x'] = self.pose.A[0,3]
        data['pos_y'] = self.pose.A[1,3]
        data['pos_z'] = self.pose.A[2,3]

        quats = sm.base.r2q(self.pose.A[:3,:3])
        data['quat_w'] = quats[0]
        data['quat_x'] = quats[1]
        data['quat_y'] = quats[2]
        data['quat_z'] = quats[3]

        return data


    def set_pose(self, **kwargs) -> bool:
        """Sets the pose data using a set of keyword arguments. The translational and rotational parts
        are independent. If the translational or rotational part is not provided the corresponding
        elements will be set to the default (0, 0, 0 for translational, and 0 degrees for rotational).

        Args:
            **kwargs:
                **pos_x** (*float*): the x compononent for translation.
                
                **pos_y** (*float*): the y compononent for translation.
                
                **pos_z** (*float*): the y compononent for translation.
                
                **quat_w** (*float*): the w compononent for rotation.
                
                **quat_x** (*float*): the x compononent for rotation.
                
                **quat_y** (*float*): the y compononent for rotation.
                
                **quat_z** (*float*): the z compononent for rotation.

        Returns:
            bool: true if the pose was successfully set.
        """

        if all([x in kwargs for x in PoseComponents.FULL]):
            self.pose = sm.SE3([kwargs.get(x) for x in PoseComponents.POS_ONLY])
            self.pose.A[:3, :3] = sm.base.q2r([kwargs.get(x) for x in PoseComponents.ROT_ONLY])
            return True

        elif all([x in kwargs for x in PoseComponents.POS_ONLY]):
            self.pose = sm.SE3([kwargs.get(x) for x in PoseComponents.POS_ONLY])
            return True

        elif all([x in kwargs for x in PoseComponents.ROT_ONLY]):
            self.pose = sm.SE3()
            self.pose.A[:3, :3] = sm.base.q2r([kwargs.get(x) for x in PoseComponents.ROT_ONLY])
            return True

        return False


### ROBO FRAME FILE ###
class RoboFrameFile(RoboFrame, ABC):
    """An abstract base class to be used when frame data is stored as individual files (e.g., images and point clouds).
    This abstract base class provides properties for common file I/O tasks (e.g., getting the file extension or filename)

    The class is derived from the :class:`.RoboFrameBase` class.
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

    The class is derived from the :class:`.RoboFrameFile` class.
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

        Args:
            **kwargs:
                **image_format** (:class:`.ImageFormat`, optional): the format for the returned image data. Defaults to ImageFormat.OPENCV
                
                **colour** (*bool*, optional): used to force the image colour type. Set to True to read in colour, False for grayscale, or
                None if the colour should be automatically determined. Defaults to True.

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

    The class is derived from the :class:`.RoboFrameFile` class.
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

        Args:
            **kwargs: None

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

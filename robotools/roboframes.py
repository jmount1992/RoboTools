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

import pathlib
from typing import List

from robotools.defines import ImageFormat
from robotools.file_utils import extension_from_filepath, read_image, read_pointcloud

###############
### CLASSES ###
###############


### ROBO FRAME BASE ###
class RoboFrameBase():

    def __init__(self, frame_id: int) -> None:
        self.frame_id = int(frame_id)


### ROBO FRAME CSV ###
class RoboFrameCSV(RoboFrameBase):

    def __init__(self, frame_id: int, fields: List = [], values: List = []) -> None:
        super().__init__(frame_id)

        if len(fields) != len(values):
            raise ValueError("The number of fields and values must be equal.")

        for field, val in zip(fields, values):
            self.add_data(field.lower(), val)

    
    def add_data(self, name: str, value) -> None:
        setattr(self, name.lower(), value)

### ROBO FRAME FILE ###
class RoboFrameFile(RoboFrameBase):

    def __init__(self, frame_id: int, filepath: pathlib.Path) -> None:
        super().__init__(frame_id)
        self.filepath = pathlib.Path(filepath)

     # Filename properties
    @property
    def filestem(self) -> str:
        """The filestem for the frame (e.g., '/<path-to-file>/<filestem>.<extension>').

        Returns:
            str: the filestem for the frame
        """
        return self.filepath.stem
    
    @property
    def filename(self) -> str:
        """The filename for the frame which is the filestem plus the extension (e.g., '/<path-to-file>/<filestem>.<extension>').

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


### ROBO FRAME IMAGE ###
class RoboFrameImage(RoboFrameFile):

    def __init__(self, frame_id: int, filepath: pathlib.Path) -> None:
        super().__init__(frame_id, filepath)

    
    def read(self, image_format: ImageFormat=ImageFormat.OPENCV, colour: bool=True):
        return read_image(self.filepath, image_format, colour)


### ROBO FRAME IMAGE ###
class RoboFrameImage(RoboFrameFile):

    def __init__(self, frame_id: int, filepath: pathlib.Path) -> None:
        super().__init__(frame_id, filepath)

    
    def read(self):
        return read_pointcloud(self.filepath)

########################
### PUBLIC FUNCTIONS ###
########################



#########################
### PRIVATE FUNCTIONS ###
#########################

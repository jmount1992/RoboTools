RoboTools Semantics
====================

Nomenclature
------------

- **Extension**: the file extension without the period.
- **Filename**: the name of the file with the extension.
- **Filepath**: the absolute path to the file including the filename.
- **Filestem**: the name of the file without the extension.
- **Frame**: the data collected by a sensor at a given point in time including any metadata (e.g., timestamp, pose, transform, etc.).
- **Rootpath**: the absolute path to the file without the filename.


File Naming Conventions for Individual Data Frames
--------------------------------------------------

Sets containing data stored as individual files (e.g., images and point clouds) must follow one of three conventions. The file naming options for individual frames are:

- `<id-number>.<extension>`;
- `<prefix>_<id-number>.<extension>`; or
- `<prefix>_<user-data>_<id-number>.<extension>`.

In addition, the rules and definitions below will be followed:

- the ID number must be an integer;
- the prefix cannot contain any underscores; and
- the extension is considered to be the string preceeding the final period.

Valid examples include:

- 00001.png
- image_001.png
- image_some_user-note_001.png


CSV File Requirements
---------------------

The use of CSV files to store data is common. RoboTools places some restrictions on certain CSV headers if you wish to have access to the full suite of tools. The restrictions are as follows:

- All CSV headers will be converted to lowercase within the software. For example, if a header called `My_Field` is present, it will be accessible using `my_field` within the RoboTools software.
- For time stamp features to be available, the CSV file must use the header `timestamp`. Spaces, hypens, or underscores are not allowed. Capitalization does not matter as all headers are converted to lowercase.
- For pose features to be available, the CSV file must contain the headers `pos_x`, `pos_y`, `pos_z`, `quat_w`, `quat_x`, `quat_y`, `quat_z` (order does not matter). The translational and rotaional components are decoupled. For example, if only `pos_x`, `pos_y`, and `pos_z` are present, the RoboTools software will assume a 0 degree rotational component. 
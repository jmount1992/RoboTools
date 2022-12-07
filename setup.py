# Created with QCR's code template tool: https://github.com/qcr/code_templates

from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='robotools',
      version='0.0.1',
      author='James Mount',
      author_email='jmount1992@gmail.com',
      url='https://github.com/jmount1992/RoboTools',
      description='A Python Package containing utilities to help undertake robotic projects',
      license_files=['LICENSE.txt'],
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      install_requires=[],
      classifiers=(
          "Development Status :: 4 - Beta",
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ))
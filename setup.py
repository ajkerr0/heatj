"""


@author: Alex Kerr
"""

from setuptools import find_packages, setup

setup(name="heatj",
      version="0.1.0",
      description="Calculate the steady-state heat current through a harmonic lattice.",
      author="Alex Kerr",
      author_email="ajkerr0@gmail.com",
      url="https://github.com/ajkerr0/heatj",
      packages=find_packages(),
      install_requires=[
      'numpy',
      ],
      )
      
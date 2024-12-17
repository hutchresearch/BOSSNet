"""
Utility functions for the bossnet inference pipeline.

MIT License
Copyright (c) 2024 hutchresearch

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from enum import Enum
from yaml import safe_load


def open_yaml(path: str) -> dict:
    with open(path) as f:
        return safe_load(f)

class Stats:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, Stats(**value))
            else:
                setattr(self, key, value)

class DataSource(str, Enum):
    """
    An enumeration that represents the data source of a dataset.

    Enumeration members:
    - BOSS: represents the data source for BOSS (Baryon Oscillation Spectroscopic Survey).
    - APOGEE: represents the data source for APOGEE (Apache Point Observatory Galactic Evolution Experiment)
    - LAMOSTDR7: represents the data source for LAMOST (Large Sky Area Multi-Object Fiber Spectroscopic Telescope).
    - LAMOSTDR8: represents the data source for LAMOST (Large Sky Area Multi-Object Fiber Spectroscopic Telescope).
    - GAIAGDR3: represents the data source for high-res GAIA
    - GAIAXP 
    """
    BOSS="boss"
    APOGEE="apogee"
    LAMOSTDR7="lamost_dr7"
    LAMOSTDR8="lamost_dr8"
    GAIAXP="gaia_xp"
    GAIAGDR3="gaia_gdr3"
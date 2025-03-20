"""
Static Program Slicer
--------------------
A static program analysis tool for backward slicing and LLM comparison.
"""

__version__ = '0.1.0'
__author__ = 'Your Name'

# Import main components to make them available at package level
from custom_slicer import StaticSlicer, generate_static_slice, SlicingResult

# Define what's exported when using 'from static_slicer import *'
__all__ = ['StaticSlicer', 'generate_static_slice', 'SlicingResult']
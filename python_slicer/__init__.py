"""
Python Slicer Module
-------------------
Advanced utilities for static program slicing.
"""

# Import classes to make them available at module level
from .slicer import (
    SliceVisualizer,
    AliasAnalyzer,
    ControlDependenceAnalyzer,
    enhance_pdg_with_control_deps
)

# Define exports
__all__ = [
    'SliceVisualizer',
    'AliasAnalyzer',
    'ControlDependenceAnalyzer',
    'enhance_pdg_with_control_deps'
]
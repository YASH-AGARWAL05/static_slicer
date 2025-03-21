# Static Program Slicer

An advanced static program backslicing tool for Python code analysis and explanation comparison.

## Overview

This tool implements backward static program slicing for Python code, enabling precise analysis of code dependencies and program structure. It generates slices that show exactly which statements affect a specific variable at a given line.

Key features:
- Abstract Syntax Tree (AST) parsing of Python code
- Control Flow Graph (CFG) and Program Dependence Graph (PDG) construction
- Backward static slicing based on data and control dependencies
- Visualization of program dependencies and slices
- Comparison framework for LLM-generated explanations

## Installation

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/static_slicer.git
   cd static_slicer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Generating a Static Slice

```bash
python main.py slice <python_file> <line_number> <variable_name> [options]
```

Example:
```bash
python main.py slice examples/buggy_sorting.py 15 sorted_list --visualize
```

This will:
- Generate a backward static slice for the variable `sorted_list` at line 15
- Save the slice text to the results folder
- Create and save a PDG visualization (with the `--visualize` flag)

### Comparing with LLM Explanations

```bash
python main.py compare <python_file> <line_number> <variable_name> --llm <model_name>
```

Example:
```bash
python main.py compare examples/logic_error.py 10 result --llm gpt-4
```

This will:
- Generate a static slice for the specified criterion
- Get an explanation from the specified LLM
- Create a comparison report and save it to the results folder

### Batch Processing

```bash
python main.py batch <directory> [--criteria <criteria_file>]
```

Example:
```bash
python main.py batch examples
```

This will:
- Process all Python files listed in the criteria.csv file
- Generate slices and visualizations for each criterion
- Save results and statistics to the results folder

## Project Structure

```
static_slicer/
├── custom_slicer.py         # Core static slicer implementation
├── main.py                  # Command-line interface
├── python_slicer/           # Additional slicing utilities
│   ├── __init__.py
│   └── slicer.py            # Enhanced visualization and analysis tools
├── examples/                # Example Python files with bugs
│   ├── buggy_sorting.py     # Bubble sort with off-by-one error
│   ├── logic_error.py       # Discount calculation bug
│   ├── off_by_one.py        # Index error example
│   └── criteria.csv         # Slicing criteria for batch processing
├── tests/                   # Unit tests
│   ├── __init__.py
│   └── test_slicer.py       # Test cases for the slicer
└── results/                 # Output directory for slices and visualizations
```

## Visualization Outputs

The tool generates two types of visualization outputs:

1. **PDG Graphs**: Shows the Program Dependence Graph with control and data dependencies.
2. **Slice Text**: Shows the code statements included in the slice.

All visualizations are saved to the `results/` directory by default.

## LLM Integration

To use LLM comparison features:

1. Install the appropriate API library:
   ```bash
   pip install openai  # For OpenAI models
   # or
   pip install anthropic  # For Anthropic Claude
   ```

2. Set up your API keys as environment variables:
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   # or
   export ANTHROPIC_API_KEY="your_api_key_here"
   ```

3. Run the compare command as shown in the usage section.

## Extending the Slicer

To add support for more Python features:

1. Extend the AST parsing in `custom_slicer.py`
2. Add new node types to the CFG/PDG construction methods
3. Update the `_collect_variables` method to handle additional variable usage patterns

## Running Tests

```bash
python -m unittest discover tests
```

## Dissertation Context

This tool was developed as part of a dissertation research project comparing static program slicing with LLM-generated explanations for code comprehension and debugging. The key research questions addressed include:

1. How do program slicing-based explanations compare to LLM-generated explanations?
2. When should developers use static slicing vs. LLMs for program comprehension?

## License

[MIT License](LICENSE)

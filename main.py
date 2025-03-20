#!/usr/bin/env python3
"""
Static Slicer - Main Program
----------------------------
An advanced static program backslicing tool for Python code analysis.

This tool is designed to:
1. Parse Python code and build a Program Dependence Graph (PDG)
2. Compute static backward slices based on a criterion
3. Compare slices with LLM-generated explanations for debugging and program comprehension

Usage:
    python main.py slice <filename> <line_number> <variable_name>
    python main.py compare <filename> <line_number> <variable_name> --llm <model_name>

Example:
    python main.py slice examples/buggy_sorting.py 15 sorted_list
    python main.py compare examples/logic_error.py 10 result --llm gpt-4
"""

import os
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from custom_slicer import StaticSlicer, generate_static_slice


def load_file(file_path):
    """Load a file and return its content."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        sys.exit(1)


def visualize_slice_stats(slice_results, output_dir='results'):
    """Generate visualizations for slicing statistics."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract statistics
    data = []
    for filename, result in slice_results.items():
        data.append({
            'filename': Path(filename).stem,
            'slice_size': len(result.slice_lines),
            'original_size': len(result.original_code.split('\n')),
            'reduction': 1 - (len(result.slice_lines) / len(result.original_code.split('\n')))
        })

    df = pd.DataFrame(data)

    # Create reduction percentage bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(df['filename'], df['reduction'] * 100)
    plt.xlabel('File')
    plt.ylabel('Code Reduction (%)')
    plt.title('Static Slice Reduction Percentage')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'slice_reduction.png'))

    # Create comparison of original vs. slice size
    plt.figure(figsize=(12, 6))
    width = 0.35
    x = range(len(df))
    plt.bar([i - width / 2 for i in x], df['original_size'], width, label='Original Lines')
    plt.bar([i + width / 2 for i in x], df['slice_size'], width, label='Slice Lines')
    plt.xlabel('File')
    plt.ylabel('Line Count')
    plt.title('Original Code vs. Slice Size')
    plt.xticks(x, df['filename'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'size_comparison.png'))

    # Save raw data
    df.to_csv(os.path.join(output_dir, 'slice_stats.csv'), index=False)

    print(f"Visualizations saved to {output_dir}")


def generate_llm_explanation(code, line_num, variable_name, model="gpt-4"):
    """
    Generate an explanation using an LLM.

    Note: This is a placeholder. In your dissertation, you would connect
    to actual LLM APIs like OpenAI, Anthropic, etc.
    """
    try:
        import openai

        # Example implementation with OpenAI
        prompt = f"""
        Analyze this Python code and explain what affects the variable '{variable_name}' at line {line_num}:

        ```python
        {code}
        ```

        Focus on:
        1. Which statements directly or indirectly affect this variable?
        2. What dependencies exist?
        3. How would you debug an issue with this variable?
        """

        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a code analysis assistant that explains code dependencies."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    except ImportError:
        return f"""
        [This is a placeholder for an LLM-generated explanation]

        To implement actual LLM integration:
        1. Install the appropriate API library (openai, anthropic, etc.)
        2. Set up your API keys
        3. Replace this function with actual API calls

        For your dissertation, you would compare this explanation with the static slice
        to evaluate correctness, comprehensibility, and usefulness.
        """


def process_directory(directory, criterion_file='criteria.csv'):
    """Process all Python files in a directory using criteria from a CSV file."""
    criteria_path = os.path.join(directory, criterion_file)

    if not os.path.exists(criteria_path):
        print(f"Error: Criteria file {criteria_path} not found")
        return

    try:
        criteria_df = pd.read_csv(criteria_path)
    except Exception as e:
        print(f"Error reading criteria file: {e}")
        return

    slice_results = {}

    for _, row in criteria_df.iterrows():
        filename = row['filename']
        line_num = int(row['line_num'])
        variable = row['variable']

        file_path = os.path.join(directory, filename)
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found, skipping")
            continue

        print(f"Processing {filename}, line {line_num}, variable '{variable}'")
        code = load_file(file_path)
        result = generate_static_slice(code, line_num, variable)
        slice_results[filename] = result

        # Save individual result
        output_dir = os.path.join(directory, 'results')
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, f"slice_{Path(filename).stem}_{variable}.txt"), 'w') as f:
            f.write(str(result))

    # Generate visualizations
    visualize_slice_stats(slice_results, os.path.join(directory, 'results'))


def llm_slice_comparison(code, line_num, variable_name, model="gpt-4"):
    """Compare static slice with LLM explanation."""
    # Generate static slice
    slice_result = generate_static_slice(code, line_num, variable_name)

    # Generate LLM explanation
    llm_explanation = generate_llm_explanation(code, line_num, variable_name, model)

    # Create comparison report
    report = f"""
    # Static Slice vs. LLM Explanation Comparison

    ## Slicing Criterion
    - Line: {line_num}
    - Variable: '{variable_name}'

    ## Static Slice
    ```
    {slice_result.slice_code}
    ```

    ## LLM Explanation ({model})
    {llm_explanation}

    ## Comparison Notes
    [Add your analysis here comparing the two approaches]

    ### Evaluation Metrics
    - Correctness: [1-5]
    - Comprehensibility: [1-5] 
    - Usefulness for debugging: [1-5]
    """

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Static Slicer - Advanced program backslicing tool for Python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Slice command
    slice_parser = subparsers.add_parser('slice', help='Generate a static slice')
    slice_parser.add_argument('filename', help='Python file to analyze')
    slice_parser.add_argument('line', type=int, help='Line number for slicing criterion')
    slice_parser.add_argument('variable', help='Variable name for slicing criterion')
    slice_parser.add_argument('--output', '-o', help='Output file for slice result')
    slice_parser.add_argument('--visualize', '-v', action='store_true', help='Visualize the PDG')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare slice with LLM explanation')
    compare_parser.add_argument('filename', help='Python file to analyze')
    compare_parser.add_argument('line', type=int, help='Line number for slicing criterion')
    compare_parser.add_argument('variable', help='Variable name for slicing criterion')
    compare_parser.add_argument('--llm', default='gpt-4', help='LLM model to use for comparison')
    compare_parser.add_argument('--output', '-o', help='Output file for comparison report')

    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Process a directory of Python files')
    batch_parser.add_argument('directory', help='Directory containing Python files')
    batch_parser.add_argument('--criteria', default='criteria.csv',
                              help='CSV file with slicing criteria (filename,line_num,variable)')

    args = parser.parse_args()

    if args.command == 'slice':
        # Generate slice for a single file
        code = load_file(args.filename)
        result = generate_static_slice(code, args.line, args.variable)

        # Create results directory if it doesn't exist
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)

        # Determine output file name
        if args.output:
            output_text_file = args.output
        else:
            # Create a default filename based on the input file and criterion
            base_filename = Path(args.filename).stem
            output_text_file = os.path.join(results_dir, f"slice_{base_filename}_{args.variable}_line{args.line}.txt")

        # Save the slice text result
        with open(output_text_file, 'w') as f:
            f.write(str(result))

        # Also print to console
        print(result)

        if args.visualize:
            slicer = StaticSlicer()
            slicer.parse_code(code)
            slicer._build_cfg()
            slicer._build_pdg()
            slicer.compute_slice((args.line, args.variable))

            # Always create a visualization output path
            vis_output_file = None
            if args.output:
                vis_output_file = f"{os.path.splitext(args.output)[0]}.png"
            else:
                # Create default visualization filename
                base_filename = Path(args.filename).stem
                vis_output_file = os.path.join(results_dir, f"pdg_{base_filename}_{args.variable}_line{args.line}.png")

            # Always save the visualization and also display it
            print(f"Saving visualization to {vis_output_file}")
            slicer.visualize_pdg(vis_output_file)

    elif args.command == 'compare':
        # Compare slice with LLM explanation
        code = load_file(args.filename)
        report = llm_slice_comparison(code, args.line, args.variable, args.llm)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
        else:
            print(report)

    elif args.command == 'batch':
        # Process a directory of files
        process_directory(args.directory, args.criteria)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
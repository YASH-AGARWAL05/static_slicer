"""
Static Program Slicer for Python Code Analysis.

This module provides tools for:
1. Parsing Python code into an Abstract Syntax Tree (AST)
2. Building a Control Flow Graph (CFG)
3. Building a Program Dependence Graph (PDG)
4. Computing static backward slices based on a given criterion
"""

import ast
import astor
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional, Union
import os


class Variable:
    """Class representing a variable in the code with its scope information."""

    def __init__(self, name: str, lineno: int, scope: str = "global"):
        self.name = name
        self.lineno = lineno
        self.scope = scope

    def __eq__(self, other):
        if not isinstance(other, Variable):
            return False
        return (self.name == other.name and
                self.scope == other.scope)

    def __hash__(self):
        return hash((self.name, self.scope))

    def __repr__(self):
        return f"Variable(name='{self.name}', lineno={self.lineno}, scope='{self.scope}')"


class StaticSlicer:
    """Advanced static program slicer for Python code."""

    def __init__(self):
        self.ast_tree = None
        self.source_code = None
        self.line_to_node = {}
        self.cfg = nx.DiGraph()
        self.pdg = nx.DiGraph()
        self.variables = {}
        self.current_scope = "global"
        self.current_function = None
        self.line_to_code = {}
        self.slicing_criterion = None

    def parse_code(self, source_code: str) -> ast.AST:
        """Parse Python source code into an AST."""
        self.source_code = source_code
        self.ast_tree = ast.parse(source_code)
        self._map_line_to_nodes(self.ast_tree)
        return self.ast_tree

    def _map_line_to_nodes(self, tree: ast.AST) -> None:
        """Maps line numbers to their corresponding AST nodes."""
        for node in ast.walk(tree):
            if hasattr(node, 'lineno'):
                if node.lineno not in self.line_to_node:
                    self.line_to_node[node.lineno] = []
                self.line_to_node[node.lineno].append(node)

                # Store original source code for each line
                if node.lineno not in self.line_to_code:
                    try:
                        self.line_to_code[node.lineno] = astor.to_source(node).strip()
                    except:
                        # Fallback for complex nodes
                        self.line_to_code[node.lineno] = f"# Complex statement at line {node.lineno}"

    def _build_cfg(self) -> nx.DiGraph:
        """Build a control flow graph from the AST."""
        self.cfg = nx.DiGraph()

        # Traverse the AST and add nodes/edges for control flow
        self._traverse_ast_for_cfg(self.ast_tree)

        return self.cfg

    def _traverse_ast_for_cfg(self, node: ast.AST, parent=None) -> None:
        """Traverse AST to build the CFG."""
        if not hasattr(node, 'lineno'):
            return

        # Add node to CFG
        self.cfg.add_node(node.lineno, ast_node=node)

        # Connect to parent if exists
        if parent is not None and hasattr(parent, 'lineno'):
            self.cfg.add_edge(parent.lineno, node.lineno)

        # Recursive traversal based on node type
        if isinstance(node, ast.If):
            # If statement has true and false branches
            self._traverse_ast_for_cfg(node.test, node)

            for stmt in node.body:
                self._traverse_ast_for_cfg(stmt, node)

            for stmt in node.orelse:
                self._traverse_ast_for_cfg(stmt, node)

        elif isinstance(node, ast.For) or isinstance(node, ast.While):
            # Loops
            if hasattr(node, 'target'):
                self._traverse_ast_for_cfg(node.target, node)
            if hasattr(node, 'iter'):
                self._traverse_ast_for_cfg(node.iter, node)
            if hasattr(node, 'test'):
                self._traverse_ast_for_cfg(node.test, node)

            for stmt in node.body:
                self._traverse_ast_for_cfg(stmt, node)

            for stmt in node.orelse:
                self._traverse_ast_for_cfg(stmt, node)

        elif isinstance(node, ast.FunctionDef):
            # Save current function context
            old_function = self.current_function
            old_scope = self.current_scope

            # Update scope information
            self.current_function = node.name
            self.current_scope = node.name

            for stmt in node.body:
                self._traverse_ast_for_cfg(stmt, node)

            # Restore scope
            self.current_function = old_function
            self.current_scope = old_scope

        elif isinstance(node, ast.Call):
            # Function calls
            self._traverse_ast_for_cfg(node.func, node)

            for arg in node.args:
                self._traverse_ast_for_cfg(arg, node)

        elif isinstance(node, ast.Assign):
            # Assignment
            for target in node.targets:
                self._traverse_ast_for_cfg(target, node)

            self._traverse_ast_for_cfg(node.value, node)

        elif isinstance(node, ast.Attribute):
            self._traverse_ast_for_cfg(node.value, node)

        elif isinstance(node, ast.BinOp):
            self._traverse_ast_for_cfg(node.left, node)
            self._traverse_ast_for_cfg(node.right, node)

        elif isinstance(node, ast.Compare):
            self._traverse_ast_for_cfg(node.left, node)

            for comparator in node.comparators:
                self._traverse_ast_for_cfg(comparator, node)

    def _build_pdg(self) -> nx.DiGraph:
        """Build a program dependence graph from the CFG and data flow."""
        self.pdg = nx.DiGraph()

        # First, add all nodes from CFG to PDG
        for node in self.cfg.nodes():
            self.pdg.add_node(node, ast_node=self.cfg.nodes[node]['ast_node'])

        # Add control dependencies
        self._add_control_dependencies()

        # Add data dependencies
        self._add_data_dependencies()

        return self.pdg

    def _add_control_dependencies(self) -> None:
        """Add control dependency edges to the PDG."""
        for node in self.cfg.nodes():
            ast_node = self.cfg.nodes[node]['ast_node']

            # Check if this is a control statement
            if isinstance(ast_node, (ast.If, ast.For, ast.While)):
                # The body of these statements is control-dependent on the statement
                if isinstance(ast_node, ast.If):
                    # For If, add control dependencies to body and orelse
                    for stmt in ast_node.body:
                        if hasattr(stmt, 'lineno'):
                            self.pdg.add_edge(node, stmt.lineno, type='control')

                    for stmt in ast_node.orelse:
                        if hasattr(stmt, 'lineno'):
                            self.pdg.add_edge(node, stmt.lineno, type='control')

                else:  # For or While loop
                    for stmt in ast_node.body:
                        if hasattr(stmt, 'lineno'):
                            self.pdg.add_edge(node, stmt.lineno, type='control')

    def _collect_variables(self) -> None:
        """Collect all variable definitions and usages."""
        self.variables = {}

        class VariableCollector(ast.NodeVisitor):
            def __init__(self, slicer):
                self.slicer = slicer
                self.current_scope = "global"

            def visit_FunctionDef(self, node):
                old_scope = self.current_scope
                self.current_scope = node.name

                # Function parameters are defined variables
                for arg in node.args.args:
                    var = Variable(arg.arg, node.lineno, self.current_scope)
                    if var not in self.slicer.variables:
                        self.slicer.variables[var] = {'defined_at': [], 'used_at': []}
                    self.slicer.variables[var]['defined_at'].append(node.lineno)

                # Visit function body
                for stmt in node.body:
                    self.visit(stmt)

                self.current_scope = old_scope

            def visit_Assign(self, node):
                # Visit right side first to collect used variables
                self.visit(node.value)

                # Now visit targets (assignment defines variables)
                for target in node.targets:
                    self._process_target(target, node.lineno)

            def _process_target(self, target, lineno):
                if isinstance(target, ast.Name):
                    var = Variable(target.id, lineno, self.current_scope)
                    if var not in self.slicer.variables:
                        self.slicer.variables[var] = {'defined_at': [], 'used_at': []}
                    self.slicer.variables[var]['defined_at'].append(lineno)
                elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
                    for elt in target.elts:
                        self._process_target(elt, lineno)
                elif isinstance(target, ast.Attribute):
                    # Handle attributes like obj.attr = value
                    self.visit(target.value)  # Visit the object part

            def visit_Name(self, node):
                # Names in loading context are variable usages
                if isinstance(node.ctx, ast.Load):
                    var = Variable(node.id, node.lineno, self.current_scope)
                    if var not in self.slicer.variables:
                        self.slicer.variables[var] = {'defined_at': [], 'used_at': []}
                    self.slicer.variables[var]['used_at'].append(node.lineno)

        collector = VariableCollector(self)
        collector.visit(self.ast_tree)

    def _add_data_dependencies(self) -> None:
        """Add data dependency edges to the PDG."""
        # First collect all variables
        self._collect_variables()

        # For each variable usage, add edges from its definition to usage
        for var, info in self.variables.items():
            for def_line in info['defined_at']:
                for use_line in info['used_at']:
                    # Only add edge if definition comes before usage
                    if def_line < use_line:
                        # Check if there's no other definition in between
                        if not any(def_line < other_def < use_line
                                   for other_def in info['defined_at']):
                            self.pdg.add_edge(def_line, use_line, type='data', variable=var.name)

    def compute_slice(self, criterion: Tuple[int, str]) -> Set[int]:
        """
        Compute a backward slice based on the given criterion.

        Args:
            criterion: A tuple (line_number, variable_name) specifying the slicing criterion

        Returns:
            A set of line numbers that are part of the slice
        """
        self.slicing_criterion = criterion
        line_num, var_name = criterion

        # Make sure PDG is built
        if not self.pdg:
            if not self.ast_tree:
                raise ValueError("No code has been parsed yet")
            self._build_cfg()
            self._build_pdg()

        # Find all Variable objects with this name
        variables = [v for v in self.variables.keys() if v.name == var_name]

        if not variables:
            return set()

        # Start with the criterion line
        slice_lines = {line_num}
        worklist = [line_num]

        # Perform backward reachability analysis on PDG
        while worklist:
            current = worklist.pop(0)

            # Add all predecessors in PDG
            for pred, _, edge_data in self.pdg.in_edges(current, data=True):
                if pred not in slice_lines:
                    slice_lines.add(pred)
                    worklist.append(pred)

        return slice_lines

    def get_slice_code(self, slice_lines: Set[int]) -> str:
        """
        Convert a set of line numbers to the corresponding source code.

        Args:
            slice_lines: Set of line numbers in the slice

        Returns:
            String containing the source code of the slice
        """
        # Sort lines to maintain program order
        sorted_lines = sorted(slice_lines)

        # Get source code for each line
        slice_code = []
        for line in sorted_lines:
            if line in self.line_to_code:
                slice_code.append(f"{line}: {self.line_to_code[line]}")

        return "\n".join(slice_code)

    def visualize_pdg(self, output_file: str = None) -> None:
        """
        Visualize the Program Dependence Graph.

        Args:
            output_file: If provided, save the visualization to this file
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 10))

        # Create a readable graph
        g = nx.DiGraph()

        # Add nodes with labels
        for node in self.pdg.nodes():
            label = f"Line {node}"
            if node in self.line_to_code:
                code_snippet = self.line_to_code[node].replace('\n', ' ')
                if len(code_snippet) > 30:
                    code_snippet = code_snippet[:27] + "..."
                label += f": {code_snippet}"
            g.add_node(node, label=label)

        # Add edges with different styles for control and data dependencies
        for src, dst, data in self.pdg.edges(data=True):
            edge_type = data.get('type', '')
            variable = data.get('variable', '')

            if edge_type == 'control':
                g.add_edge(src, dst, style='dashed', color='red', label='control')
            elif edge_type == 'data':
                g.add_edge(src, dst, style='solid', color='blue', label=f"data: {variable}")

        # Highlight slicing criterion if exists
        pos = nx.spring_layout(g, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(g, pos, node_color='lightblue', node_size=700)

        # Highlight slicing criterion node if it exists
        if self.slicing_criterion:
            criterion_line = self.slicing_criterion[0]
            if criterion_line in g.nodes():
                nx.draw_networkx_nodes(g, pos,
                                       nodelist=[criterion_line],
                                       node_color='yellow',
                                       node_size=800)

        # Draw edges
        for edge in g.edges(data=True):
            nx.draw_networkx_edges(
                g, pos,
                edgelist=[(edge[0], edge[1])],
                edge_color=edge[2].get('color', 'black'),
                style=edge[2].get('style', 'solid'),
                width=1.5
            )

        # Draw labels
        nx.draw_networkx_labels(g, pos, labels={n: g.nodes[n]['label'] for n in g.nodes()})

        # Draw edge labels
        edge_labels = {(src, dst): data.get('label', '') for src, dst, data in g.edges(data=True)}
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=8)

        plt.title("Program Dependence Graph (PDG)")
        plt.axis("off")

        # Save the file if an output path is provided
        if output_file:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            plt.savefig(output_file, bbox_inches='tight')
            print(f"Visualization saved to {output_file}")

        # Always show the plot (this won't block in non-interactive environments)
        plt.show()


class SlicingResult:
    """Class to hold and format slicing results."""

    def __init__(self, slice_lines: Set[int], slice_code: str, criterion: Tuple[int, str], original_code: str):
        self.slice_lines = slice_lines
        self.slice_code = slice_code
        self.criterion = criterion
        self.original_code = original_code

    def __str__(self) -> str:
        """String representation of the slicing result."""
        result = [
            f"=== Slicing Criterion: Line {self.criterion[0]}, Variable '{self.criterion[1]}' ===",
            f"Lines in slice: {sorted(self.slice_lines)}",
            "",
            "=== Slice Code ===",
            self.slice_code
        ]
        return "\n".join(result)

    def to_html(self) -> str:
        """Generate HTML representation with syntax highlighting."""
        from pygments import highlight
        from pygments.lexers import PythonLexer
        from pygments.formatters import HtmlFormatter

        # Highlight the slice code
        highlighted_slice = highlight(self.slice_code, PythonLexer(), HtmlFormatter())

        # Create HTML structure
        html = f"""
        <div class="slicing-result">
            <h3>Slicing Criterion: Line {self.criterion[0]}, Variable '{self.criterion[1]}'</h3>
            <p>Lines in slice: {sorted(self.slice_lines)}</p>
            <h4>Slice Code:</h4>
            {highlighted_slice}
        </div>
        """
        return html


def generate_static_slice(source_code: str, line_num: int, variable_name: str) -> SlicingResult:
    """
    Generate a static backward slice for the given source code, line number, and variable.

    Args:
        source_code: Python source code to analyze
        line_num: Line number for the slicing criterion
        variable_name: Variable name for the slicing criterion

    Returns:
        SlicingResult object containing the slice
    """
    slicer = StaticSlicer()
    slicer.parse_code(source_code)
    slicer._build_cfg()
    slicer._build_pdg()

    slice_lines = slicer.compute_slice((line_num, variable_name))
    slice_code = slicer.get_slice_code(slice_lines)

    return SlicingResult(slice_lines, slice_code, (line_num, variable_name), source_code)


if __name__ == "__main__":
    # Example usage
    example_code = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total

def find_average(numbers):
    if not numbers:
        return 0
    total = calculate_sum(numbers)
    count = len(numbers)
    average = total / count
    return average

data = [1, 2, 3, 4, 5]
result = find_average(data)
print(f"The average is: {result}")
"""

    # Generate slice for the 'average' variable at line 13
    slice_result = generate_static_slice(example_code, 13, 'average')
    print(slice_result)
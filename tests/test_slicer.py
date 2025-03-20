"""
Advanced static program slicing utilities.

This module contains additional tools and utilities for static program slicing,
extending the functionality of the main StaticSlicer class.
"""

import ast
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional


class SliceVisualizer:
    """Utilities for visualizing program slices and dependence graphs."""

    @staticmethod
    def highlight_slice_in_code(source_code: str, slice_lines: Set[int]) -> str:
        """
        Highlight sliced lines in the original source code.

        Args:
            source_code: Original Python source code
            slice_lines: Set of line numbers in the slice

        Returns:
            Highlighted code as string with HTML formatting
        """
        from pygments import highlight
        from pygments.lexers import PythonLexer
        from pygments.formatters import HtmlFormatter

        # Split code into lines
        lines = source_code.split('\n')

        # Add HTML markers for highlighted lines
        highlighted_lines = []
        for i, line in enumerate(lines):
            # Line numbers are 1-based, but list indices are 0-based
            line_num = i + 1

            if line_num in slice_lines:
                highlighted_lines.append(f'<span class="highlight">{line}</span>')
            else:
                highlighted_lines.append(line)

        # Rejoin the code
        highlighted_code = '\n'.join(highlighted_lines)

        # Apply syntax highlighting
        html = highlight(highlighted_code, PythonLexer(), HtmlFormatter())

        # Add custom CSS for highlighting
        css = """
        <style>
        .highlight {
            background-color: #ffff99;
            display: block;
        }
        </style>
        """

        return css + html


class AliasAnalyzer:
    """Analyze variable aliases in Python code."""

    def __init__(self, ast_tree: ast.AST):
        self.ast_tree = ast_tree
        self.aliases = {}
        self._analyze()

    def _analyze(self):
        """Find variable aliases in the code."""

        class AliasVisitor(ast.NodeVisitor):
            def __init__(self):
                self.aliases = {}
                self.current_scope = "global"

            def visit_FunctionDef(self, node):
                old_scope = self.current_scope
                self.current_scope = node.name

                # Visit function body
                for stmt in node.body:
                    self.visit(stmt)

                self.current_scope = old_scope

            def visit_Assign(self, node):
                # Handle simple aliasing: x = y
                if (len(node.targets) == 1 and
                        isinstance(node.targets[0], ast.Name) and
                        isinstance(node.value, ast.Name)):

                    alias = node.targets[0].id
                    original = node.value.id

                    scope_key = f"{self.current_scope}"
                    if scope_key not in self.aliases:
                        self.aliases[scope_key] = {}

                    self.aliases[scope_key][alias] = original

                # Continue normal visit
                self.generic_visit(node)

        visitor = AliasVisitor()
        visitor.visit(self.ast_tree)
        self.aliases = visitor.aliases

    def get_alias_chain(self, var_name: str, scope: str = "global") -> List[str]:
        """
        Get the chain of aliases for a variable.

        Args:
            var_name: Variable name to check
            scope: Scope to search in

        Returns:
            List of variable names in the alias chain
        """
        if scope not in self.aliases or var_name not in self.aliases[scope]:
            return [var_name]

        result = [var_name]
        current = var_name

        # Follow the chain of aliases
        while scope in self.aliases and current in self.aliases[scope]:
            current = self.aliases[scope][current]
            # Avoid circular references
            if current in result:
                break
            result.append(current)

        return result


class ControlDependenceAnalyzer:
    """Advanced control dependency analysis for PDG construction."""

    def __init__(self, ast_tree: ast.AST, cfg: nx.DiGraph):
        self.ast_tree = ast_tree
        self.cfg = cfg
        self.dominators = {}
        self.post_dominators = {}

    def compute_control_dependencies(self) -> Dict[int, Set[int]]:
        """
        Compute control dependencies using post-dominance frontiers.

        Returns:
            Dictionary mapping line numbers to sets of line numbers they control
        """
        # Compute post-dominators
        self._compute_post_dominators()

        # Compute control dependencies
        control_deps = {}

        for node in self.cfg.nodes():
            control_deps[node] = set()

            # For each successor of the node
            for succ in self.cfg.successors(node):
                # If node does not post-dominate successor, node controls successor
                if node not in self.post_dominators.get(succ, set()):
                    control_deps[node].add(succ)

        return control_deps

    def _compute_post_dominators(self):
        """Compute post-dominators for all nodes in the CFG."""
        # Create a reversed CFG for post-dominators
        reverse_cfg = self.cfg.reverse()

        # Find exit nodes (nodes with no successors in original CFG)
        exit_nodes = [n for n in self.cfg.nodes() if self.cfg.out_degree(n) == 0]

        if not exit_nodes:
            # If no natural exit nodes, use nodes with highest line numbers
            exit_nodes = [max(self.cfg.nodes())]

        # Initialize post-dominators
        self.post_dominators = {}
        all_nodes = set(self.cfg.nodes())

        for node in all_nodes:
            if node in exit_nodes:
                self.post_dominators[node] = {node}
            else:
                self.post_dominators[node] = all_nodes

        # Iteratively refine post-dominators
        changed = True
        while changed:
            changed = False

            for node in all_nodes:
                if node in exit_nodes:
                    continue

                # Compute intersection of post-dominators of all successors
                successors = list(reverse_cfg.successors(node))
                if not successors:
                    continue

                new_post_doms = set.intersection(
                    *[self.post_dominators[s] for s in successors]
                )
                new_post_doms.add(node)

                if self.post_dominators[node] != new_post_doms:
                    self.post_dominators[node] = new_post_doms
                    changed = True

        return self.post_dominators


def enhance_pdg_with_control_deps(slicer, ast_tree, cfg, pdg):
    """
    Enhance a PDG with more accurate control dependencies.

    Args:
        slicer: The StaticSlicer instance
        ast_tree: The AST tree
        cfg: The Control Flow Graph
        pdg: The Program Dependence Graph to enhance

    Returns:
        Enhanced PDG with more precise control dependencies
    """
    # Create analyzer
    cd_analyzer = ControlDependenceAnalyzer(ast_tree, cfg)

    # Compute control dependencies
    control_deps = cd_analyzer.compute_control_dependencies()

    # Add control dependency edges to PDG
    for node, controlled_nodes in control_deps.items():
        for controlled in controlled_nodes:
            pdg.add_edge(node, controlled, type='control')

    return pdg
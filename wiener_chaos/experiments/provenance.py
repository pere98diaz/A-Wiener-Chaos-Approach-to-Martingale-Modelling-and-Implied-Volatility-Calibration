"""Fingerprint executable Python source for saved-run provenance checks.

Fingerprints retain imports, definitions, constants, and statements while
ignoring comments, formatting, and docstrings. This lets documentation edits
preserve valid checkpoints. Numerical changes still invalidate them; relocating
or restructuring code requires an explicit audit of any historical results.
"""

import ast
import hashlib
from pathlib import Path


class _WithoutDocstrings(ast.NodeTransformer):
    def generic_visit(self, node):
        node = super().generic_visit(node)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:]
        return node


def source_fingerprint(path, *, stop_before=None):
    """Hash executable syntax, optionally stopping before a plotting-only function."""
    tree = ast.parse(Path(path).read_text())
    if stop_before is not None:
        for index, node in enumerate(tree.body):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == stop_before:
                tree.body = tree.body[:index]
                break
        else:
            raise ValueError(f"No function {stop_before!r} in {path}")
    canonical = ast.dump(_WithoutDocstrings().visit(tree), include_attributes=False)
    return "python-ast-v1:" + hashlib.sha256(canonical.encode()).hexdigest()

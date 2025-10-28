"""Parser for the textual DAG DSL."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from .ast import BindingDecl, GraphDecl, NodeDecl, OutputDecl, ParameterDecl


class DSLParseError(SyntaxError):
    """Raised when the DSL parser encounters invalid syntax."""


class _DSLParser:
    def __init__(self, text: str):
        self.lines = text.splitlines()
        self.index = 0
        self.length = len(self.lines)

    def parse(self) -> List[GraphDecl]:
        graphs: List[GraphDecl] = []
        while self.index < self.length:
            if not self._current_stripped():
                self.index += 1
                continue
            graph = self._parse_graph()
            graphs.append(graph)
        return graphs

    def _parse_graph(self) -> GraphDecl:
        line = self._current_line()
        stripped = line.strip()
        if self._indent_of(line) != 0:
            raise self._error("Expected GRAPH declaration at indentation level 0")

        if not stripped.startswith("GRAPH "):
            if stripped.lower().startswith("graph "):
                raise self._error("Use uppercase keyword 'GRAPH'")
            raise self._error("Expected 'GRAPH <name>:'")

        if not stripped.endswith(":"):
            raise self._error("Graph declaration must end with ':'")

        payload = stripped[len("GRAPH ") : -1].strip()
        if not payload:
            raise self._error("Graph declaration missing name")
        metadata: Mapping[str, Any] = {}
        if "(" in payload:
            name_part, meta_part = payload.split("(", 1)
            name = name_part.strip()
            metadata = self._parse_metadata(meta_part.rstrip(")"), self.index + 1)
        else:
            name = payload
        self._validate_identifier(name, "graph name")
        graph = GraphDecl(name=name, metadata=metadata)
        self.index += 1
        block_indent = self._expect_indented_block()
        while self.index < self.length:
            line = self._current_line()
            stripped = line.strip()
            if not stripped:
                self.index += 1
                continue
            indent = self._indent_of(line)
            if indent < block_indent:
                break
            if indent > block_indent:
                raise self._error("Inconsistent indentation")
            if stripped.startswith("PARAMETER "):
                statement = self._collect_statement_with_continuations(block_indent)
                self._parse_parameter(graph, statement)
            elif stripped.startswith("INPUT "):
                statement = self._collect_statement_with_continuations(block_indent)
                self._parse_input(graph, statement)
            elif stripped.startswith("OUTPUT "):
                statement = self._collect_statement_with_continuations(block_indent)
                self._parse_output(graph, statement)
            elif stripped.lower().startswith("parameter "):
                raise self._error("Use uppercase keyword 'PARAMETER'")
            elif stripped.lower().startswith("input "):
                raise self._error("Use uppercase keyword 'INPUT'")
            elif stripped.lower().startswith("output "):
                raise self._error("Use uppercase keyword 'OUTPUT'")
            else:
                self._parse_node(graph, stripped)
            self.index += 1
        return graph

    def _collect_statement_with_continuations(self, block_indent: int) -> str:
        """Collect the current line plus any indented continuations."""
        parts: List[str] = [self._current_line().strip()]
        lookahead = self.index + 1
        while lookahead < self.length:
            candidate = self.lines[lookahead]
            stripped = candidate.strip()
            if not stripped:
                break
            indent = self._indent_of(candidate)
            if indent <= block_indent:
                break
            parts.append(stripped)
            lookahead += 1
        self.index = lookahead - 1
        return " ".join(parts)

    def _parse_input(self, graph: GraphDecl, stripped: str) -> None:
        payload = stripped[len("INPUT ") :].strip()
        if not payload:
            raise self._error("Input declaration requires names")
        for alias in payload.replace(",", " ").split():
            self._validate_identifier(alias, "input")
            if alias in graph.inputs:
                raise self._error(f"Duplicate input '{alias}'")
            graph.inputs.append(alias)

    def _parse_output(self, graph: GraphDecl, stripped: str) -> None:
        payload = stripped[len("OUTPUT ") :].strip()
        if not payload:
            raise self._error("Output declaration requires a source")
        if "=" in payload:
            alias_part, source_part = payload.split("=", 1)
            alias = alias_part.strip()
            source = source_part.strip()
            self._validate_identifier(alias, "output alias")
        else:
            source = payload.strip()
            alias = source.split(".")[-1] if "." in source else source
            self._validate_identifier(alias, "output alias")
        if "." not in source:
            source = f"{source}._return"
        graph.outputs.append(OutputDecl(alias=alias, source=source, line=self.index + 1))

    def _parse_node(self, graph: GraphDecl, stripped: str) -> None:
        if "=" not in stripped:
            raise self._error("Expected assignment 'node = operator[...]'")
        lhs, rhs = stripped.split("=", 1)
        name = lhs.strip()
        self._validate_identifier(name, "node name")
        if any(node.name == name for node in graph.nodes):
            raise self._error(f"Duplicate node '{name}'")
        operator_expr, bindings_expr = self._split_operator_bindings(rhs.strip())
        bindings = self._parse_bindings(bindings_expr)
        graph.nodes.append(
            NodeDecl(
                name=name,
                operator_expr=operator_expr,
                bindings=bindings,
                metadata={},
                line=self.index + 1,
            )
        )

    def _split_operator_bindings(self, rhs: str) -> Tuple[str, str]:
        if "[" not in rhs:
            return self._normalise_operator_expr(rhs), ""
        if not rhs.endswith("]"):
            raise self._error("Bindings must end with ']'")
        operator_expr, bindings_expr = rhs[:-1].split("[", 1)
        return self._normalise_operator_expr(operator_expr.strip()), bindings_expr.strip()

    @staticmethod
    def _normalise_operator_expr(expr: str) -> str:
        return expr.replace("(*)", "()")

    def _parse_bindings(self, bindings_str: str) -> List[BindingDecl]:
        if not bindings_str:
            return []
        bindings: List[BindingDecl] = []
        for part in bindings_str.split(","):
            piece = part.strip()
            if not piece:
                continue
            if "=" not in piece:
                raise self._error("Binding must use 'port=source' syntax")
            port_part, value_part = piece.split("=", 1)
            port = port_part.strip()
            self._validate_identifier(port, "port")
            value = value_part.strip()
            default_expr = None
            source = None
            if ":" in value:
                source_section, default_section = value.split(":", 1)
                source = source_section.strip() or None
                default_expr = default_section.strip() or None
            else:
                source = value or None
            if source is None and default_expr is None:
                raise self._error("Binding must specify a source or default")
            bindings.append(
                BindingDecl(
                    port=port,
                    source=source,
                    default_expr=default_expr,
                    line=self.index + 1,
                )
            )
        return bindings

    def _parse_parameter(self, graph: GraphDecl, stripped: str) -> None:
        payload = stripped[len("PARAMETER ") :].strip()
        if not payload:
            raise self._error("Parameter declaration requires names")
        for piece in payload.split(","):
            part = piece.strip()
            if not part:
                continue
            if "=" in part:
                name_part, default_part = part.split("=", 1)
                name = name_part.strip()
                default_expr = default_part.strip() or None
            else:
                name = part
                default_expr = None
            self._validate_identifier(name, "parameter")
            if any(param.name == name for param in graph.parameters):
                raise self._error(f"Duplicate parameter '{name}'")
            graph.parameters.append(
                ParameterDecl(
                    name=name,
                    default_expr=default_expr,
                    line=self.index + 1,
                )
            )

    def _parse_metadata(self, meta_str: str, line: int) -> Mapping[str, Any]:
        meta: Dict[str, Any] = {}
        for chunk in meta_str.split(","):
            piece = chunk.strip()
            if not piece:
                continue
            if "=" not in piece:
                raise DSLParseError(f"Invalid metadata pair '{piece}' (line {line})")
            key, value = piece.split("=", 1)
            key = key.strip()
            if not key:
                raise DSLParseError(f"Metadata key cannot be empty (line {line})")
            meta[key] = value.strip()
        return meta

    def _expect_indented_block(self) -> int:
        while self.index < self.length:
            line = self._current_line()
            if not line.strip():
                self.index += 1
                continue
            indent = self._indent_of(line)
            if indent == 0:
                raise self._error("Expected indented block")
            return indent
        raise self._error("Unexpected EOF while parsing graph body")

    def _current_line(self) -> str:
        return self.lines[self.index]

    def _current_stripped(self) -> str:
        return self.lines[self.index].strip()

    @staticmethod
    def _indent_of(line: str) -> int:
        return len(line) - len(line.lstrip(" "))

    def _validate_identifier(self, name: str, label: str) -> None:
        if not name:
            raise self._error(f"{label} cannot be empty")
        if not (name[0].isalpha() or name[0] == "_") or not all(
            ch.isalnum() or ch == "_" for ch in name
        ):
            raise self._error(f"Invalid {label} '{name}'")

    def _error(self, message: str) -> DSLParseError:
        return DSLParseError(f"{message} (line {self.index + 1})")


__all__ = ["DSLParseError", "_DSLParser"]

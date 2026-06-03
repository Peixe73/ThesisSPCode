from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Union, Optional, Any
import xml.etree.ElementTree as ET


# IR (Internal Representation)

@dataclass
class Restriction:
    """
    Represents OWL restriction like:
        ∃hasBorder.Red
        ∀hasShape.Triangle
    """
    property: str
    filler: str
    kind: str  # "some", "all", "not"


@dataclass
class ClassExpr:
    """
    Represents a named OWL class with a logical definition.
    """
    name: str
    expression: List[Union[str, Restriction, "ClassExpr"]]


# OWL PARSER

class OWLParser:
    def __init__(self, owl_path: str):
        self.tree = ET.parse(owl_path)
        self.root = self.tree.getroot()

        self.ns = {
            "owl": "http://www.w3.org/2002/07/owl#",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        }

        # final mapping: class_name -> ClassExpr
        self.classes: Dict[str, ClassExpr] = {}

    # Public API
    def parse(self) -> Dict[str, ClassExpr]:
        for cls in self.root.findall(".//owl:Class", self.ns):
            about = cls.attrib.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about")
            if not about:
                continue

            name = about.split("#")[-1]

            eq = cls.find("owl:equivalentClass", self.ns)
            if eq is None:
                continue

            expr = self._parse_expression(eq)

            self.classes[name] = ClassExpr(
                name=name,
                expression=expr
            )

        return self.classes

    # Expression parsing
    def _parse_expression(self, node):
        result = []

        inter = node.find(".//owl:intersectionOf", self.ns)
        if inter is None:
            return result

        # IMPORTANT: OWL RDF collection handling
        for item in inter:
            parsed = self._parse_node(item)
            if parsed is not None:
                result.append(parsed)

        return result

    def _parse_node(self, node) -> Optional[Union[str, Restriction, ClassExpr]]:

        # Restriction node
        restriction = node.find("owl:Restriction", self.ns)
        if restriction is not None:

            prop = restriction.find(".//owl:onProperty", self.ns)
            some = restriction.find(".//owl:someValuesFrom", self.ns)
            all_ = restriction.find(".//owl:allValuesFrom", self.ns)

            prop_name = self._extract_resource(prop)

            if some is not None:
                filler = self._extract_resource(some)
                return Restriction(prop_name, filler, "some")

            if all_ is not None:
                filler = self._extract_resource(all_)
                return Restriction(prop_name, filler, "all")

        # Complement
        comp = node.find("owl:complementOf", self.ns)
        if comp is not None:
            inner = self._parse_node(comp)
            if isinstance(inner, Restriction):
                inner.kind = "not"
            return inner

        # Union
        union = node.find("owl:unionOf", self.ns)
        if union is not None:
            parts = []
            for u in union:
                p = self._parse_node(u)
                if p is not None:
                    parts.append(p)
            return " or ".join(self._render(p) for p in parts)

        # Named class
        about = node.attrib.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about")
        if about:
            return self._short(about)

        return None

    # Helpers
    def _extract_resource(self, node) -> str:
        if node is None:
            return "Thing"

        resource = node.attrib.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource")
        if resource:
            return self._short(resource)

        return "Thing"

    def _short(self, uri: str) -> str:
        return uri.split("#")[-1] if "#" in uri else uri


# MANCHSTER GENERATOR

class ManchesterGenerator:
    def __init__(self, parsed_classes: Dict[str, ClassExpr]):
        self.classes = parsed_classes

    # Public API
    def to_manchester(self, cls: str, negate: bool = False) -> str:
        prefix = "__input__ Type: "
        neg = "not " if negate else ""

        if cls not in self.classes:
            raise ValueError(f"Unknown OWL class: {cls}")

        expr = self.classes[cls].expression

        body = " and ".join(self._render(e) for e in expr)

        return f"{prefix}{neg}({body})"

    # Rendering logic
    def _render(self, e: Any) -> str:

        # Restriction
        if isinstance(e, Restriction):
            if e.kind == "some":
                return f"(∃{e.property}.{e.filler})"
            elif e.kind == "all":
                return f"(∀{e.property}.{e.filler})"
            elif e.kind == "not":
                return f"not (∃{e.property}.{e.filler})"

        # String union already rendered
        if isinstance(e, str):
            return e

        # Fallback
        return str(e)


# CONVENIENCE WRAPPER

class OWLManchesterCompiler:
    """
    One-stop interface:
    OWL file -> Manchester generator
    """

    def __init__(self, owl_path: str):
        parser = OWLParser(owl_path)
        parsed = parser.parse()
        self.generator = ManchesterGenerator(parsed)

    def to_manchester(self, cls: str, negate: bool = False) -> str:
        return self.generator.to_manchester(cls, negate)


# EXAMPLE USAGE

if __name__ == "__main__":
    compiler = OWLManchesterCompiler("ontologies/gtsrb.owl")

    print(compiler.to_manchester("B1"))
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

try:
    from jsonschema import Draft202012Validator, SchemaError
except ImportError:  # pragma: no cover
    Draft202012Validator = None  # type: ignore[assignment]
    SchemaError = Exception  # type: ignore[assignment,misc]

from anthropic_protocol.models import ToolDefinition, ToolInvocation
from agent_bridge.tool_parser import ParsedToolCall


class ToolRegistryError(ValueError):
    """Base error for tool-registry operations."""


class ToolNotFoundError(ToolRegistryError):
    """The model requested a tool Claude Code did not advertise."""


class ToolInputValidationError(ToolRegistryError):
    """The model supplied tool input that does not satisfy its JSON Schema."""


class ToolSchemaError(ToolRegistryError):
    """A supplied tool schema is itself invalid."""


@dataclass(slots=True)
class RegisteredTool:
    definition: ToolDefinition
    validator: Any = field(repr=False, default=None)


@dataclass(slots=True)
class ToolRegistry:
    """Authoritative registry built from the tools Claude Code supplied."""

    _tools: dict[str, RegisteredTool] = field(default_factory=dict)

    def register(self, definition: ToolDefinition) -> RegisteredTool:
        name = definition.name.strip()
        if not name:
            raise ToolRegistryError("tool name must be non-empty")
        if name in self._tools:
            raise ToolRegistryError(f"duplicate tool definition: {name}")

        validator = None
        schema = definition.input_schema or {}
        if Draft202012Validator is None:
            raise ToolSchemaError("jsonschema dependency is required for tool validation")
        try:
            validator = Draft202012Validator(schema)
            validator.check_schema(schema)
        except SchemaError as exc:
            raise ToolSchemaError(f"invalid schema for tool {name}: {exc.message}") from exc

        registered = RegisteredTool(
            definition=ToolDefinition(
                name=name,
                description=definition.description,
                input_schema=schema,
                raw=definition.raw,
            ),
            validator=validator,
        )
        self._tools[name] = registered
        return registered

    def register_many(self, definitions: Iterable[ToolDefinition]) -> None:
        for definition in definitions:
            self.register(definition)

    def get(self, name: str) -> RegisteredTool:
        registered = self._tools.get(name.strip())
        if registered is None:
            raise ToolNotFoundError(f"unknown tool: {name}")
        return registered

    def names(self) -> list[str]:
        return list(self._tools)

    def validate_input(self, name: str, input_data: dict[str, Any]) -> None:
        registered = self.get(name)
        errors = sorted(registered.validator.iter_errors(input_data), key=lambda e: list(e.path))
        if not errors:
            return
        details = "; ".join(self._format_error(error) for error in errors[:5])
        raise ToolInputValidationError(f"invalid input for tool {name}: {details}")

    def build_invocation(self, parsed: ParsedToolCall, tool_id: str) -> ToolInvocation:
        self.validate_input(parsed.name, parsed.input)
        return ToolInvocation(
            id=tool_id,
            name=parsed.name,
            input=parsed.input,
            status="pending",
        )

    @staticmethod
    def _format_error(error: Any) -> str:
        path = "".join(f"[{part!r}]" if isinstance(part, int) else f".{part}" for part in error.path)
        location = path[1:] if path.startswith(".") else path
        return f"{location or '<root>'}: {error.message}"

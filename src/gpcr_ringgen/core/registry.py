"""A deliberately small dictionary-backed component registry."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Generic, TypeVar

from gpcr_ringgen.core.exceptions import RegistryError

T = TypeVar("T")


class Registry(Generic[T]):
    """Store named components with duplicate and missing-name checks."""

    def __init__(self, category: str) -> None:
        self.category = category
        self._items: dict[str, T] = {}

    def register(self, name: str, item: T) -> None:
        """Register an item under a non-empty unique name."""
        normalized = name.strip()
        if not normalized:
            raise RegistryError("Registry names must not be empty.")
        if normalized in self._items:
            raise RegistryError(
                f"{self.category!r} component {normalized!r} is already registered."
            )
        self._items[normalized] = item

    def get(self, name: str) -> T:
        """Return a registered item or raise a project-specific error."""
        try:
            return self._items[name]
        except KeyError as exc:
            raise RegistryError(
                f"Unknown {self.category!r} component: {name!r}."
            ) from exc

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def names(self) -> tuple[str, ...]:
        """Return names in registration order."""
        return tuple(self._items)

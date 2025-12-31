from enum import Enum

try:
    from typing import Self
except ImportError:  # pragma: <3.11 cover
    from typing_extensions import Self

from packaging.specifiers import SpecifierSet
from packaging.version import Version


class Strategy(str, Enum):
    """Version constraint strategy for workspace dependencies."""

    PIN = "pin"
    ALLOW_PATCH_UPDATES = "allow-patch-updates"
    ALLOW_MINOR_UPDATES = "allow-minor-updates"
    ALLOW_ALL_UPDATES = "allow-all-updates"
    SEMVER = "semver"

    @classmethod
    def from_string(cls, value: str) -> Self:
        """Parse a strategy from a string value.

        Args:
            value: The string value to parse.

        Returns:
            The corresponding Strategy enum member.

        Raises:
            ValueError: If the value is not a valid strategy.
        """
        try:
            return cls(value)
        except ValueError:
            valid = ", ".join(s.value for s in cls)
            raise ValueError(f"Invalid strategy '{value}'. Must be one of: {valid}") from None

    def make_specifier(self, version: Version) -> SpecifierSet:
        """Create a PEP 440 version specifier based on the strategy.

        Args:
            version: The version to create a specifier for.

        Returns:
            A version specifier set (e.g., `>=1.2.3`, `==1.2.3`).
        """  # noqa: DOC501
        public_version = version.public

        match self:
            case Strategy.PIN:
                return SpecifierSet(f"=={public_version}")
            case Strategy.ALLOW_PATCH_UPDATES:
                next_minor = version.minor + 1
                return SpecifierSet(f">={public_version},<{version.major}.{next_minor}.0")
            case Strategy.ALLOW_MINOR_UPDATES:
                next_major = version.major + 1
                return SpecifierSet(f">={public_version},<{next_major}.0.0")
            case Strategy.ALLOW_ALL_UPDATES:
                return SpecifierSet(f">={public_version}")
            case Strategy.SEMVER:
                if version.major == 0:
                    return Strategy.ALLOW_PATCH_UPDATES.make_specifier(version)
                return Strategy.ALLOW_MINOR_UPDATES.make_specifier(version)
            case _:  # pragma: no cover
                raise AssertionError(f"Unhandled strategy: {self}")

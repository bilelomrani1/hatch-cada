import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from hatch_cada.strategy import Strategy


class TestFromString:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("pin", Strategy.PIN),
            ("allow-patch-updates", Strategy.ALLOW_PATCH_UPDATES),
            ("allow-minor-updates", Strategy.ALLOW_MINOR_UPDATES),
            ("allow-all-updates", Strategy.ALLOW_ALL_UPDATES),
            ("semver", Strategy.SEMVER),
        ],
    )
    def test_parses_valid_strategy(self, value: str, expected: Strategy) -> None:
        assert Strategy.from_string(value) == expected

    def test_raises_for_invalid_strategy(self) -> None:
        with pytest.raises(ValueError, match="Invalid strategy 'invalid'"):
            Strategy.from_string("invalid")

    def test_error_lists_valid_options(self) -> None:
        with pytest.raises(ValueError, match="allow-all-updates"):
            Strategy.from_string("wrong")


class TestMakeSpecifier:
    @pytest.mark.parametrize(
        ("strategy", "version_str", "expected"),
        [
            (Strategy.PIN, "1.2.3", "==1.2.3"),
            (Strategy.ALLOW_PATCH_UPDATES, "1.2.3", ">=1.2.3,<1.3.0"),
            (Strategy.ALLOW_MINOR_UPDATES, "1.2.3", ">=1.2.3,<2.0.0"),
            (Strategy.ALLOW_ALL_UPDATES, "1.2.3", ">=1.2.3"),
            (Strategy.SEMVER, "0.2.3", ">=0.2.3,<0.3.0"),
            (Strategy.SEMVER, "1.2.3", ">=1.2.3,<2.0.0"),
        ],
    )
    def test_creates_specifier(self, strategy: Strategy, version_str: str, expected: str) -> None:
        version = Version(version_str)
        assert strategy.make_specifier(version) == SpecifierSet(expected)

    def test_strips_local_version(self) -> None:
        version = Version("1.2.3+local")
        assert Strategy.ALLOW_ALL_UPDATES.make_specifier(version) == SpecifierSet(">=1.2.3")

    def test_preserves_dev_version(self) -> None:
        version = Version("1.2.3.dev1")
        assert Strategy.ALLOW_ALL_UPDATES.make_specifier(version) == SpecifierSet(">=1.2.3.dev1")

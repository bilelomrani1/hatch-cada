# Changelog

<!-- version list -->

## [v1.0.3](https://github.com/bilelomrani1/hatch-cada/compare/v1.0.2...v1.0.3) (2026-02-24)
### Bug Fixes

- Handle circular workspace dependencies by [@bilelomrani1](https://github.com/bilelomrani1) ([71e6012](https://github.com/bilelomrani1/hatch-cada/commit/71e601214caf8b492108362351826baa0ce98f56)) in [#45](https://github.com/bilelomrani1/hatch-cada/pull/45)

    Fixed `RecursionError` when workspace packages with dynamic versions have circular dependencies and both use hatch-cada.


## [v1.0.2](https://github.com/bilelomrani1/hatch-cada/compare/v1.0.1...v1.0.2) (2026-02-23)
### Bug Fixes

- Handle directory path dependencies in lockfile by [@bilelomrani1](https://github.com/bilelomrani1) ([6c7a206](https://github.com/bilelomrani1/hatch-cada/commit/6c7a206c22793e8b495bf20ef7c12e955bf55221)) in [#43](https://github.com/bilelomrani1/hatch-cada/pull/43)

    Support `source.directory` (file:// path dependencies) in the lockfile parser. Previously, packages declared as `file://` path dependencies would raise a `KeyError` if their version was omitted from the lockfile.


## [v1.0.1](https://github.com/bilelomrani1/hatch-cada/compare/v1.0.0...v1.0.1) (2026-01-12)
### Bug Fixes

- Handle missing packages in lockfile during `uv add` by [@bilelomrani1](https://github.com/bilelomrani1) ([6c06f84](https://github.com/bilelomrani1/hatch-cada/commit/6c06f841320a0270e57c141870de941d88043055)) in [#19](https://github.com/bilelomrani1/hatch-cada/pull/19)

    Fixed `KeyError` when running `uv add`. The metadata hook now gracefully handles packages not
       yet present in the lockfile.


# [v1.0.0](https://github.com/bilelomrani1/hatch-cada/compare/v0.1.0...v1.0.0) (2025-12-31)
### Features

- Add support for optional dependencies by [@bilelomrani1](https://github.com/bilelomrani1) ([5f5cf9c](https://github.com/bilelomrani1/hatch-cada/commit/5f5cf9c13157f1fe5bfc8dda10ae9892bc627cca)) in [#2](https://github.com/bilelomrani1/hatch-cada/pull/2)

    Add support for `project.optional-dependencies`. Workspace dependencies in optional dependency groups are now rewritten with version specifiers during build.


## [v0.1.0](https://github.com/bilelomrani1/hatch-cada/releases/tag/v0.1.0) (2025-12-31)
### Features

- First release by [@bilelomrani1](https://github.com/bilelomrani1) ([a665b50](https://github.com/bilelomrani1/hatch-cada/commit/a665b5044f43f3d7f84d03ad7f483c0501a44311))

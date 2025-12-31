# Cada

[uv workspaces](https://docs.astral.sh/uv/concepts/workspaces/) are great for developing multiple Python packages together in a monorepo. But uv doesn't currently solve how to *build and publish* packages that depend on each other.

Cada is a simple [hatchling](https://pypi.org/project/hatchling/) build plugin for uv workspaces. It rewrites workspace dependencies with version constraints at build time, so each package can be published independently.

## Table of contents  <!-- omit in toc -->

- [Why?](#why)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Comparison with other plugins](#comparison-with-other-plugins)
  - [Una](#una)
  - [hatch-dependency-coversion](#hatch-dependency-coversion)
  - [uv-dynamic-versioning](#uv-dynamic-versioning)
- [Development](#development)
- [License](#license)

## Why?

In a monorepo, you shouldn't have to manually track version constraints for internal dependencies. Each package defines its own version, either in `project.version` or derived from VCS tags. At build time, Cada reads these versions and generates constraints based on a compatibility strategy you choose. You focus on code, not version bookkeeping.

Suppose you have a monorepo managed by a uv workspace with internal libraries that depend on each other:

```toml
[project]
name = "my-client"
dependencies = ["my-core"]

[tool.uv.sources]
my-core = { workspace = true }
```

This works great for local development. But when you build `my-client`, the resulting wheel depends on `my-core` with no version constraint, which might not be desirable.

Cada solves this by generating version constraints at build time:

```
my-core-2.0.0.whl
my-client-1.0.0.whl   # depends on my-core>=2.0.0
```

Each package keeps its own version and release cycle. Users only install what they need. Update `my-core` and consumers of `my-client` get it transitively ie. no new `my-client` release is required.

## Quickstart

Add Cada to your build dependencies in each package that has workspace dependencies:

```toml
[build-system]
requires = ["hatchling", "hatch-cada"]
build-backend = "hatchling.build"

[project]
name = "my-client"
dependencies = ["my-core"]

[tool.uv.sources]
my-core = { workspace = true }

[tool.hatch.metadata.hooks.cada]
strategy = "allow-all-updates"
```

> [!NOTE]
> Cada requires Hatchling as your build backend. The uv build backend does not support plugins yet.

> [!WARNING]
> On current versions of hatchling, metadata hooks only run when `project.dynamic` is non-empty. If you're not using dynamic fields like `dynamic = ["version"]`, add `dynamic = ["license-files"]` as a workaround. See [pypa/hatch#2153](https://github.com/pypa/hatch/issues/2153) for details.

Then build using your usual build frontend (uv, hatch, etc.):

```bash
uv build packages/my-client
```

The built wheel will have proper version constraints for all workspace dependencies.

## Configuration

Add the following to your `pyproject.toml`:

```toml
[tool.hatch.metadata.hooks.cada]
strategy = "allow-all-updates"
```

The `strategy` option controls how version constraints are generated for workspace dependencies. If `my-core` is at version `1.2.3`:

- **`allow-all-updates`** generates `my-core>=1.2.3`. This is the most common choice for libraries.

- **`allow-minor-updates`** generates `my-core>=1.2.3,<2.0.0`. Allows minor and patch updates but not major version bumps.

- **`allow-patch-updates`** generates `my-core>=1.2.3,<1.3.0`. Allows patch updates only.

- **`semver`** is version-aware: for `0.x` versions it behaves like `allow-patch-updates` (since minor bumps can be breaking in `0.x`), and for `1.x+` it behaves like `allow-minor-updates`.

- **`pin`** generates `my-core==1.2.3`. Locks to the exact current version in your workspace.

You can override the strategy for specific dependencies:

```toml
[tool.hatch.metadata.hooks.cada]
strategy = "allow-all-updates"

[tool.hatch.metadata.hooks.cada.overrides]
my-core = "pin"
my-utils = "semver"
```

With this configuration, `my-core` will be pinned to its exact version, `my-utils` will use semver-aware constraints, and all other workspace dependencies will use the default `allow-all-updates` strategy.

## Comparison with other plugins

### Una

[Una](https://github.com/carderne/una) takes a different approach: instead of publishing workspace packages independently, it bundles all workspace dependencies into a single wheel:

```
my-client-1.0.0.whl
├── my_client/
└── my_core/       # vendored inside
```

Use **Una** when you're building an **application** you deploy yourself (Docker images, Lambda functions, CLI tools). You get a single artifact with everything included ie. no external dependencies to manage at install time.

Use **Cada** when you're building **libraries for external consumers**. Each package gets its own version and release cycle, and users only install what they need.

### hatch-dependency-coversion

[hatch-dependency-coversion](https://github.com/Opentrons/hatch-plugins/tree/main/hatch-dependency-coversion) rewrites dependency versions to match the *current package's* version. This is useful for lockstep versioning where all packages in a monorepo share the same version number.

**Cada** is different: it resolves each dependency's version through hatchling's plugin system. This supports independent versioning where each package has its own release cycle, and also works with dynamic versioning plugins like [hatch-vcs](https://github.com/ofek/hatch-vcs).

### uv-dynamic-versioning

[uv-dynamic-versioning](https://github.com/ninoseki/uv-dynamic-versioning) is another hatchling plugin that can handle workspace dependencies. The key difference is in how dependencies are declared.

**uv-dynamic-versioning** requires moving dependencies out of the standard `project.dependencies` field and uses non-standard templating in the version specifier:

```toml
[project]
name = "my-client"
dynamic = ["dependencies"]

[tool.hatch.metadata.hooks.uv-dynamic-versioning]
dependencies = ["my-core=={{ version }}"]
```

**Cada** keeps standard dependencies syntax and location:

```toml
[project]
name = "my-client"
dependencies = ["my-core"]

[tool.uv.sources]
my-core = { workspace = true }

[tool.hatch.metadata.hooks.cada]
strategy = "allow-all-updates"
```

This matters because many tools in the Python ecosystem rely on the standard `project.dependencies` field:

- **Monorepo build tools** ([Nx](https://nx.dev/), [Moon](https://moonrepo.dev/), etc.) auto-detect internal dependencies to build task graphs and determine what to rebuild when code changes. Non-standard dependency locations break this detection.
- **Dependency scanners** ([pip-audit](https://github.com/pypa/pip-audit), [safety](https://github.com/pyupio/safety), etc.) won't detect vulnerabilities and won't be able to rewrite your dependencies if they are in a non-standard location.
- **Dependency update tools** such as [Dependabot](https://docs.github.com/en/code-security/dependabot) only supports PEP 621 standard locations and has no workaround. [Renovate](https://docs.renovatebot.com/modules/manager/pep621/) can be configured with [custom regex managers](https://docs.renovatebot.com/modules/manager/regex/), but this requires manual setup.
- **Import linters** ([deptry](https://github.com/fpgmaas/deptry)) can't verify imports match declared dependencies

Cada preserves compatibility with these tools out of the box by keeping your `pyproject.toml` structure standard.

## Development

Set up the development environment:

```bash
uv sync --group dev
uv run poe setup
```

Available tasks via [Poe the Poet](https://poethepoet.natn.io/):

- `uv run poe setup` - Set up development environment
- `uv run poe test` - Run tests
- `uv run poe lint` - Run linter
- `uv run poe lint:fix` - Run linter with auto-fix
- `uv run poe format` - Format code
- `uv run poe format:check` - Check code formatting
- `uv run poe typecheck` - Run type checker
- `uv run poe lock:check` - Check lockfile exists and is up to date
- `uv run poe check` - Run all checks
- `uv run poe release` - Trigger release workflow on GitHub

## License

Cada is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

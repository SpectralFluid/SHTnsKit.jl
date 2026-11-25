# API Reference

This page is generated from the docstrings of the public `SHTnsKit` module.
Optional methods become available in your Julia session when you load the
packages that activate their extension.

## Calling conventions

- Allocating transforms use `operation(cfg, input...)`.
- In-place transforms place outputs before inputs, for example
  `analysis!(plan, alm_out, field)` and
  `synthesis!(plan, field_out, alm)`.
- `analysis(CPU(), ...)` and `analysis(GPU(), ...)` request strict typed device
  dispatch; ordinary calls infer the backend from array storage.
- Dense coefficients have shape `(cfg.lmax + 1, cfg.mmax + 1)`. Entries with
  `l < m` are unused.
- Batch APIs add a third, field-index dimension.
- A configuration's `norm`, `cs_phase`, `real_norm`, and `robert_form` options
  apply at every public transform boundary.

See [Migrating to v2.0](../migration.md) for removed compatibility signatures
and numerical changes.

## Extension activation

```@eval
using Markdown, TOML, SHTnsKit
extensions = TOML.parsefile(joinpath(pkgdir(SHTnsKit), "Project.toml"))["extensions"]
rows = ["| Extension | Trigger packages |", "|:---|:---|"]
for name in sort!(collect(keys(extensions)))
    triggers = extensions[name]
    packages = triggers isa String ? [triggers] : triggers
    push!(rows, "| `$name` | " * join(("`$pkg`" for pkg in packages), ", ") * " |")
end
Markdown.parse(join(rows, "\n"))
```

Metal loads KernelAbstractions as a dependency, so `using SHTnsKit, Metal`
activates its extension. For setup and supported operations see
[GPU Acceleration](../gpu.md). [Package Architecture](../architecture.md)
describes each extension's role.

## Public API

```@autodocs
Modules = [SHTnsKit]
Private = false
Order = [:module, :constant, :type, :function, :macro]
```

## Configuration lifecycle

`SHTConfig` is managed by Julia's garbage collector. `destroy_config(cfg)` is a
no-op retained for API symmetry; callers do not need a `try`/`finally` teardown
pattern.

## Related guides

- [Quick Start](../quickstart.md)
- [Grid Types](../grids.md)
- [Normalization and Phase](../norms.md)
- [Advanced Usage](../advanced.md)
- [Distributed Computing](../distributed.md)

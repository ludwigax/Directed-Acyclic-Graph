# DAG Core Refactor Plan

## 1. Objectives
- Untangle `dag/node.py` into cohesive modules that separate *spec authoring* concerns from *runtime execution*.
- Provide fine-grained execution units suitable for parallel scheduling by flattening nested graphs before runtime creation.
- Introduce clear lifecycle separation: specification -> template shell -> instantiated runtime.
- Maintain ergonomics for DSL authors while enabling programmatic graph construction without DSL.

## 2. Guiding Principles
- **Single Responsibility:** each module owns one concern (inspection, specification, execution, tooling).
- **No Incidental Instantiation:** specs and templates never eagerly create operator instances; instances exist only inside runtimes.
- **Composable Nodes:** every runtime node represents one callable with resolved defaults. No hidden subgraphs inside nodes.
- **Metadata First:** templates expose metadata/configuration APIs even before instantiation so UIs and tooling can inspect graphs safely.
- **Backward Compatibility Strategy:** keep the public facade (`register_*`, `build_graph`, DSL entry points) stable while internals are reworked.

## 3. Target Package Layout

```
dag/
|-- core/
|   |-- inspect.py          # callable introspection utilities
|   |-- ports.py            # PortDefinition, ParameterSpec, helper types
|   |-- nodes.py            # NodeTemplate, NodeShell, defaults resolution
|   |-- specs.py            # GraphSpec, NodeSpec (declarative only)
|   |-- builder.py          # Spec -> Runtime expansion & macro flattening
|   |-- runtime/
|   |   |-- plan.py         # ExecutionPlan, ExecutionContext
|   |   `-- scheduler.py    # Future parallel scheduler / cache policy
|   `-- registry.py         # OperatorRegistry and decorators
|-- dsl/
|   |-- parser.py
|   |-- program.py
|   `-- __init__.py
`-- inspect_utils/
```

`dag/__init__.py` will re-export the high-level APIs from the new modules.

## 4. Key Abstractions

### 4.1 Inspection Layer (`core.inspect`)
- Unified helpers to extract parameter names, annotations, and defaults for callables/classes without instantiating them.
- Capture existing `returns_keys` metadata; fall back to heuristics (single return value -> `_return`).
- Output feeds directly into `NodeTemplate` creation.

### 4.2 Specification Layer (`core.specs`)
- Pure data classes (`GraphSpec`, `NodeSpec`, `EdgeSpec`, etc.) describing graph topology, parameters, metadata.
- Nested graphs remain representable but are treated as macros--`GraphSpec` references can be expanded later.
- Serialization helpers (`to_dict`, `from_dict`) live here and remain backward compatible.

### 4.3 Template Shell Layer (`core.nodes`)
- `NodeTemplate`: immutable description of how to instantiate a callable, including init/call defaults and metadata.
- `NodeShell`: binds a `NodeTemplate` plus per-node config, still without touching live objects; exposes methods to:
  - merge graph-supplied defaults,
  - validate bindings against port definitions,
  - introspect ports/metadata for tooling.
- `GraphTemplate`: holds expanded `NodeShell`s and parameter defs but no instantiated runners.

### 4.4 Builder & Macro Expansion (`core.builder`)
- Consumes `GraphSpec` (possibly with nested specs) and produces a flattened `GraphTemplate`.
- Expansion rules:
  1. Inline nested `GraphSpec` by cloning their nodes, prefixing IDs, and rewiring edges/inputs/outputs.
  2. Propagate parameters/defaults outward; handle collisions with deterministic rules.
  3. Validate that after expansion every runtime node maps to a single callable.
- Only after expansion do we instantiate concrete operator objects through registry hooks.
- Exposes two stages:
  - `compile(spec) -> GraphTemplate`
  - `materialise(template, parameters) -> ExecutionPlan`

### 4.5 Runtime Layer (`core.runtime`)
- `ExecutionPlan` keeps `NodeRuntime` instances that wrap actual callables.
- `_ExecutionState` evolves into a scheduler backed by a ready-queue (supports future parallel execution).
- Node caching stays per-node but now visible through template metadata to allow disabling or custom policies.

### 4.6 Registry (`core.registry`)
- `OperatorRegistry` stores `NodeTemplate`s instead of factory callables.
- Decorators `@register_function` / `@register_class` delegate to `core.inspect` + `NodeTemplate`.
- `register_graph` compiles the provided spec into a template at registration time (macro expansion) but defers instantiation until runtime build.

## 5. DSL Integration
- `dsl.parser` continues to emit `GraphSpec`.
- During `DSLProgram.build`, instead of returning specs with embedded `GraphSpec` nodes, call into `core.builder.compile` to expand macros; result is compatible with `GraphTemplate`.
- Runtime behaviour for DSL users remains `program.build(...)` -> `build_graph(...)`.

## 6. Migration Strategy
1. **Scaffold New Modules:** create empty shells with current types migrated verbatim (no behavioural change).
2. **Move Inspection & Registry:** extract introspection helpers and registry logic; add unit tests around metadata extraction.
3. **Introduce Templates/Shells:** refactor operator instantiation path to use `NodeTemplate`/`NodeShell`. Keep current runtime class temporarily.
4. **Implement Macro Expansion:** modify builder to flatten nested graphs, updating tests/examples to reflect new node IDs.
5. **Refactor Runtime Execution:** adjust runtime to consume flattened templates. Preserve public API signature.
6. **Cleanup Legacy Code:** delete obsolete sections from old monolith, ensure imports point to new structure.
7. **Document Changes:** update README/docs to explain new architecture and any observable behaviour changes (e.g., node naming after expansion).

## 7. Compatibility & Risks
- **Node IDs:** flattening will change auto-generated node IDs; need deterministic prefixing to avoid breaking saved specs.
- **Serialization:** ensure `GraphSpec.to_dict()` remains backward compatible even if runtime uses flattened templates.
- **Performance:** macro expansion may increase graph size; scheduling improvements should offset by enabling parallelism.
- **Debugging Hooks:** migrate existing `dbg` integration so timing/caching still works with new runtime.

## 8. Validation Plan
- Unit tests for inspection, template defaults, macro expansion, runtime execution, and DSL round-trips.
- Integration test: register nested graph via DSL, build runtime, assert flattened node count and selective execution behaviour.
- Performance smoke test: verify that selective outputs execute minimal nodes with new scheduler.

## 9. Open Questions
- How to expose expanded node IDs in a user-friendly way post-flattening?
- Should templates support lazy operator loading for heavy classes?
- What is the minimal scheduler we deliver in the first pass (single-threaded queue vs. concurrent executor)?

This plan will drive the initial refactor branch. Each phase can ship independently with tests/documentation to keep the transition controlled.

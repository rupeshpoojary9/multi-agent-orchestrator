# Emergence — a self-extending multi-agent orchestrator

A working, small-scale mirror of an autonomous **multi-agent orchestrator**: a
**meta-agent** that plans a task DAG, routes each step to a **Web Agent** or an
**API Agent**, runs a **plan → execute → verify → iterate** loop that re-plans on
failure, and — the differentiator — **synthesizes a brand-new sub-agent at
runtime** when a step needs a capability no registered agent has. The whole thing
is wrapped in an integration/governance layer: typed tool contracts, an
append-only audit log, guardrails, and a cost/latency budget.

It runs **fully offline and deterministically with no API key** (a heuristic
planner/verifier/synthesizer stand in for the model), and swaps to a real
Anthropic model by setting one environment variable. Same contracts either way.

```
                       ┌─────────────────────────────────────────────┐
   natural-language    │                META-AGENT                   │
   task  ───────────►  │  plan (DAG) → route → execute → verify → ⟲  │
                       └───────┬───────────────┬──────────────┬──────┘
                               │ route          │ verify fail  │ capability gap
                        ┌──────▼──────┐   ┌──────▼──────┐  ┌────▼─────────┐
                        │  Web Agent  │   │  API Agent  │  │ SYNTHESIZE a │
                        │ (legacy UI) │   │ (typed REST)│  │  sub-agent   │
                        └─────────────┘   └─────────────┘  └──────────────┘
                     governance: audit log · allow-list · approval gate · budget
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

emergence demo "Onboard Umbrella Health across the console and the API."
emergence eval          # run the 26-task suite and print reliability metrics
emergence agents        # list registered agents
```

No API key needed. To use a real model instead of the heuristics:

```bash
pip install -e ".[llm]"
export ANTHROPIC_API_KEY=sk-...
emergence demo "Offboard Wayne Enterprises across all systems."
```

## What each pillar looks like when it runs

**Verify + recover.** The legacy web app is eventually consistent: the first
read-back after a write returns a *stale* value once. The verifier catches the
mismatch and the orchestrator retries until the read is correct.

```
$ emergence demo "In the console, set account 4 status to active."
[PASS] 4 steps  catches=1  replans=0
  [ok] set    web.set_status    x1 — executed
  [ok] read   web.read_status   x2 — status='active' == 'active'   ← caught the stale read, recovered
```

**Agents that create agents.** No foundational agent knows how to "onboard".
The meta-agent generates one at runtime — as a *recipe over vetted primitives* —
validates its schema, dry-runs it (no side effects), registers it, and routes to it.

```
$ emergence demo "Onboard Globex Corp end to end."
[PASS] 2 steps  catches=1
  [ok] wf   workflow.onboard_account   x2 — status='active' == 'active' (synth:gen_onboard_account)

synthesized agents (generated at runtime):
  • gen_onboard_account  composes=['api.set_customer_status', 'web.add_note', 'web.login', ...]
```

**Governance stops unsafe or runaway runs.**

```
$ emergence demo "Translate the quarterly report into French."
[FAIL] halted: SynthesisError: no recipe available for capability 'workflow.unknown'
```

## Metrics (offline heuristic run, 26 seeded tasks)

`emergence eval`:

| metric | value |
| --- | --- |
| end-to-end task success | **100%** |
| cross-system task success | 100% |
| synthesis-task success | 100% |
| **verifier catch rate** | **100%** (8 real failures caught / 0 leaked) |
| mean attempts / task | 1.31 |
| agents synthesized | 3 |
| latency p50 / p95 | ~0.1 / 0.2 ms |
| cost / task | $0.0000 (offline) |

Success is measured **independently** of the orchestrator's self-report: the
harness reads the mock backends directly to confirm they actually reached the
intended state (`eval/tasks.py::check_expect`). A "leak" is a run the
orchestrator called successful but whose ground truth failed — the verifier's job
is to drive that to zero. Numbers come from `emergence eval`; latency depends on
your machine and rises with the real-LLM path.

## Architecture

| Module | Responsibility |
| --- | --- |
| `planner.py` | Meta-agent: natural-language task → routed task DAG (heuristic + LLM). |
| `orchestrator.py` | The plan→execute→verify→iterate state machine; retries, re-plan, budget. |
| `verifier.py` | Checks each step's result against its success criteria (predicate + LLM). |
| `synthesizer.py` | Generates a new sub-agent when a capability is missing; validates + dry-runs it. |
| `registry.py` | Routing table + durable store of agent specs (incl. generated ones). |
| `agents/` | `APIAgent` (typed REST), `WebAgent` (legacy UI), `RecipeAgent` (synthesized). |
| `mocks/` | Seeded enterprise API (SQLite) + legacy web app (state machine). |
| `governance.py` | Audit log, allow-list + approval gate, step/re-plan/cost budget. |
| `eval/` | 26 seeded tasks with independent ground-truth checks + metrics. |

### Design decisions worth knowing

- **One contract over heterogeneous systems.** Every agent — browser or API —
  implements the same `Agent` protocol (`spec` + `run(capability, args) →
  Observation`). The orchestrator can't tell a Playwright agent from a REST agent
  at the seam. That's the integration abstraction.
- **Generated agents never invent raw capability.** A synthesized agent is a
  `RecipeAgent`: an ordered list of calls to already-registered, allow-listed
  primitives. Runtime code generation is real, but bounded — the safety story.
- **Two synthesis gates.** Before a generated agent may act it must (1) pass
  schema validation (every tool registered and allow-listed) and (2) pass a
  dry-run that executes read-only steps for real and refuses to fire writes.
- **Determinism by default.** Predicate-based verification and a recipe library
  make every pillar testable with no API key; the LLM paths are drop-in and
  change quality, not the contract.

## Driving a real browser / real services

The mocks share the method surface of their real counterparts, so the swap is
localized:

- **Real web app:** implement a `PlaywrightWebApp` with the same methods as
  `mocks/web_app.py` (login/open_account/read_status/set_status/add_note) driving
  a Playwright browser against a Flask CRUD app; construct `WebAgent` with it.
  (`pip install -e ".[web]"`)
- **Real API:** point `APIBackend` at a live REST service (same method names).
- **LangGraph backend:** the native loop in `orchestrator.py` is a plain state
  machine; `pip install -e ".[graph]"` to port it onto LangGraph unchanged.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

Covers agent dispatch and routing, DAG validity, the verifier predicates, the
verify-and-recover loop, runtime synthesis with its gates (including "dry-run has
no side effects"), every governance halt path, and the full eval suite.

## License

MIT © Rupesh Poojary

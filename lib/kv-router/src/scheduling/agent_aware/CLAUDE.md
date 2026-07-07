# AgentAware Scheduling

AgentAware is an opt-in policy overlay owned by `SchedulerQueue`. It groups tracked requests by `session_id`, keeps an active program on one worker while capacity allows, and pauses or resumes programs at the existing admission boundary. Requests without a session ID and configurations without `agent_aware` bypass the policy.

## Module Map

| File | Responsibility |
| --- | --- |
| `mod.rs` | Actor-local program state and the private `AgentAwarePolicy` surface. |
| `config.rs` | Defaults, deserialization, startup validation, and control interval. |
| `lifecycle.rs` | Request preparation, placement, admission commit or rollback, completion accounting, and explicit session release. |
| `control.rs` | Capacity calculation and the periodic force-resume, greedy-resume, and pressure-pause loop. |
| `tests.rs` | Cross-module policy behavior. Queue integration tests remain in `../queue.rs`. |

The policy is private to `crate::scheduling`. Do not add a public policy trait, service, queue, or second source of truth.

## Request Flow

```mermaid
flowchart LR
    A["SchedulingRequest"] --> B["AgentAwarePolicy::prepare"]
    B --> C["Existing PolicyQueue"]
    C --> D["AgentAwarePolicy::can_dispatch"]
    D --> E["Existing worker selector"]
    E --> F["AgentAwarePolicy::on_admitted"]
    F --> G["RequestGuard completion"]
    G --> H["AgentAwarePolicy::complete"]
    H --> I["AgentAwarePolicy::tick"]
```

1. `prepare` ignores untracked or sessionless requests. It creates or updates program state, inherits a live assignment, or selects the least-used eligible worker with enough logical capacity.
2. A paused program forces ordinary queueing. AgentAware drain uses `PolicyQueue::pop_next_skipping_blocked`; the disabled path retains the normal heap-head-only `pop_next` scan.
3. The existing selector receives `pinned_worker` when the policy has an assignment. AgentAware does not book worker state itself.
4. `on_admitted` commits the selected worker and maps the tracked request ID to its session. `on_admission_failed` rolls back provisional state.
5. `complete` adds output tokens, transitions the program from REASONING to ACTING, applies a deferred pause, and triggers the control loop through `SchedulerQueue`.
6. `end_session` removes terminal program and request mappings when the frontend supplies the final-session signal.

## Program State

- `REASONING`: a request is in flight. Pressure marks the program for pause but does not move it while the request is running.
- `ACTING`: the program is between requests and retains a logical working set. Pressure can pause it immediately.
- `paused`: no worker is assigned. A continuation remains queued until the control loop resumes the program.
- `pending_since`: records when a paused continuation began waiting and drives forced resume.
- `acting_since`: records the last transition to ACTING and drives decayed headroom only for forced-resume placement.

## Control Loop

`tick` runs no more often than `scheduler_interval_seconds` and applies this order:

1. Build per-DP-rank capacity from device KV blocks plus backend-reported retention capacity.
2. Force sessions whose queued continuation exceeded `resume_timeout_seconds` onto the worker with the most decayed headroom.
3. Resume eligible paused programs with best-fit decreasing placement while preserving `resume_hysteresis` below the pause threshold.
4. For each worker above `pause_threshold`, pause the smallest ACTING programs first and mark the smallest REASONING programs until projected use reaches `pause_target`.

Logical charge is `retained tokens * acting_token_weight + buffer_per_program` for ACTING programs and `retained tokens + buffer_per_program` for REASONING programs. Pause decisions use the conservative full charge. Only forced-resume headroom applies exponential ACTING decay.

## Invariants

- Keep all mutable program state inside the single-threaded queue actor; no locks or detached tasks.
- Preserve request eligibility, pinned-worker constraints, and data-parallel rank when selecting or resuming.
- Never pause a REASONING program immediately; set `marked_for_pause` and pause after completion.
- Never let a paused heap head block a dispatchable request behind it, but do not change normal queue eligibility semantics.
- Commit program assignment only after successful worker booking and roll it back on every admission failure path.
- Treat backend capacity as an accounting hint. The backend still owns KV admission, spill, restore, and eviction.
- Keep session-final cleanup explicit and idempotent.

## Validation

Run `cargo test -p dynamo-kv-router --lib` after policy or queue changes. Add focused tests for lifecycle transitions, capacity arithmetic, pause/resume ordering, and disabled-path behavior; keep actor-level integration tests in `../queue.rs`.

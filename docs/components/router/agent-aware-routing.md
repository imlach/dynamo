---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: AgentAware Routing
subtitle: Program-aware admission and placement for sequential agent workloads
---

AgentAware routing groups requests by session and manages their retained working sets inside the existing KV router admission actor. Its initial algorithm implements the scheduling design introduced by the [ThunderAgent paper](https://arxiv.org/abs/2602.13692) and Dynamo's standalone [ThunderAgent Program Scheduler](../../agents/thunderagent-router.md).

> [!WARNING]
> **Experimental.** Enable AgentAware routing only for agentic workloads whose sequential requests carry a stable session ID.

## Enable AgentAware Routing

Set `agent_aware` in the router policy configuration:

```yaml
agent_aware: {}
```

Start the frontend with the policy file:

```bash
python -m dynamo.frontend \
    --router-mode kv \
    --router-policy-config /etc/dynamo/agent-aware.yaml
```

Leave `--router-session-affinity-ttl-secs` unset because AgentAware routing owns session placement and may migrate a paused session. Send `X-Dynamo-Session-ID` on every request and `X-Dynamo-Session-Final: true` on the last generated turn so the router releases the session state after the response completes.

## Scheduling Behavior

AgentAware routing keeps each active session's working set on one worker while capacity allows. Requests without a session ID bypass the policy, and the existing FCFS, WSPT, Deficit Round Robin (DRR), and queue-limit behavior remains in place.

New sessions are assigned to the least-used eligible worker that has room for the estimated working set. Capacity includes device KV blocks and, when reported by SGLang HiCache, host-retained tokens.

At each control interval, the policy runs these steps in order:

1. Force placement for a paused continuation that exceeded the resume timeout.
2. Resume paused sessions with best-fit decreasing placement while headroom remains.
3. When pressure exceeds the configured threshold, pause the smallest sessions that are between requests first.
4. Mark in-flight sessions for pause when their current request completes if more pressure reduction is required.

A resumed continuation receives a queue-priority boost. If a paused session is the highest-priority entry in its policy class, the queue can dispatch a ready entry behind it without changing the normal queue path used when AgentAware routing is disabled.

## Configuration

| Field | Default | Description |
| --- | ---: | --- |
| `pause_threshold` | `0.95` | Begin working-set control above this fraction of worker capacity. |
| `pause_target` | `0.80` | Pause sessions until estimated use reaches this fraction. |
| `resume_hysteresis` | `0.10` | Reserve this fraction below the pause threshold when resuming sessions. |
| `resume_timeout_seconds` | `1800` | Force placement after a paused continuation waits this long. |
| `resume_priority_boost` | `1.0` | Add this value to a resumed continuation's queue priority. |
| `scheduler_interval_seconds` | `5.0` | Run the working-set control loop at this interval. |
| `acting_token_weight` | `1.0` | Weight retained tokens for sessions between requests. |
| `acting_decay_tau_seconds` | `1.0` | Decay time constant used only for forced-resume placement. |
| `buffer_per_program` | `100` | Reserve this many extra tokens per active session. |

For the surrounding policy-class schema and router flags, see [Configuration and Tuning](router-configuration.md). For queue arbitration details, see [Deficit Round Robin Queue Scheduling](deficit-round-robin.md).

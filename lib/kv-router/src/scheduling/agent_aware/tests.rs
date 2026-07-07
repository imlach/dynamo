// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use rustc_hash::FxHashMap;
use tokio::time::{Duration, Instant};

use super::control::worker_capacity;
use super::*;
use crate::protocols::{RoutingConstraints, WorkerId, WorkerWithDpRank};
use crate::scheduling::{OverlapSignals, ScheduleMode, SchedulingRequest};
use crate::test_utils::SimpleWorkerConfig;

fn workers() -> HashMap<WorkerId, SimpleWorkerConfig> {
    HashMap::from([(
        1,
        SimpleWorkerConfig {
            total_kv_blocks: Some(100),
            ..Default::default()
        },
    )])
}

fn request(id: &str, session_id: Option<&str>, isl_tokens: usize) -> SchedulingRequest {
    SchedulingRequest {
        mode: ScheduleMode::Tracked {
            request_id: id.to_string(),
        },
        token_seq: None,
        isl_tokens,
        lora_name: None,
        expected_output_tokens: None,
        pinned_worker: None,
        allowed_worker_ids: None,
        routing_constraints: RoutingConstraints::default(),
        router_config_override: None,
        track_prefill_tokens: true,
        priority_jump: 0.0,
        strict_priority: 0,
        policy_class: None,
        session_id: session_id.map(str::to_string),
        overlap: OverlapSignals::default(),
        shared_cache_hits: None,
        worker_loads: FxHashMap::default(),
        resp_tx: None,
    }
}

fn admit(policy: &mut AgentAwarePolicy, request: &mut SchedulingRequest, now: Instant) {
    assert!(!policy.prepare(request, &workers(), 10, now));
    let worker = request.pinned_worker.unwrap();
    policy.on_admitted(request, worker);
}

#[test]
fn assigns_new_session_and_tracks_completion() {
    let now = Instant::now();
    let mut policy = AgentAwarePolicy::new(AgentAwareConfig::default());
    let mut request = request("r1", Some("s1"), 600);

    admit(&mut policy, &mut request, now);
    assert_eq!(request.pinned_worker, Some(WorkerWithDpRank::new(1, 0)));

    policy.complete("r1", 20, now + Duration::from_secs(1));
    let program = &policy.programs["s1"];
    assert_eq!(program.status, ProgramStatus::Acting);
    assert_eq!(program.token_total, 620);
}

#[test]
fn pauses_smallest_acting_session_then_resumes_its_continuation() {
    let now = Instant::now();
    let config = AgentAwareConfig {
        pause_threshold: 0.8,
        pause_target: 0.7,
        resume_hysteresis: 0.0,
        scheduler_interval_seconds: 1.0,
        ..Default::default()
    };
    let mut policy = AgentAwarePolicy::new(config);
    for (request_id, session_id, tokens) in [("big-r1", "big", 600), ("small-r1", "small", 200)] {
        let mut request = request(request_id, Some(session_id), tokens);
        admit(&mut policy, &mut request, now);
        policy.complete(request_id, 0, now + Duration::from_millis(1));
    }

    policy.last_tick = now;
    policy.tick(&workers(), 10, now + Duration::from_secs(1));
    assert!(policy.programs["small"].paused);
    assert!(!policy.programs["big"].paused);

    let mut continuation = request("small-r2", Some("small"), 220);
    assert!(policy.prepare(
        &mut continuation,
        &workers(),
        10,
        now + Duration::from_secs(2)
    ));
    assert_eq!(continuation.priority_jump, 1.0);
    policy.end_session("big");
    policy.tick(&workers(), 10, now + Duration::from_secs(3));
    assert!(!policy.programs["small"].paused);
    policy.apply_assignment(&mut continuation);
    assert_eq!(
        continuation.pinned_worker,
        Some(WorkerWithDpRank::new(1, 0))
    );
}

#[test]
fn requests_without_session_identity_bypass_policy() {
    let mut policy = AgentAwarePolicy::new(AgentAwareConfig::default());
    let mut request = request("r1", None, 900);
    assert!(!policy.prepare(&mut request, &workers(), 10, Instant::now()));
    assert!(policy.programs.is_empty());
    assert_eq!(request.pinned_worker, None);
}

#[test]
fn capacity_includes_backend_retention_tiers() {
    let config = SimpleWorkerConfig {
        total_kv_blocks: Some(100),
        extra_kv_capacity_tokens: Some(500),
        ..Default::default()
    };
    assert_eq!(worker_capacity(&config, 10), Some(1_500));
}

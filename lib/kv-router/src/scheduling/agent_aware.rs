// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use serde::Deserialize;
use tokio::time::{Duration, Instant};

use super::policy_config::RouterPolicyConfigError;
use super::types::SchedulingRequest;
use crate::protocols::{WorkerConfigLike, WorkerId, WorkerWithDpRank};

const fn default_pause_threshold() -> f64 {
    0.95
}

const fn default_pause_target() -> f64 {
    0.80
}

const fn default_resume_hysteresis() -> f64 {
    0.10
}

const fn default_resume_timeout_seconds() -> f64 {
    1800.0
}

const fn default_resume_priority_boost() -> f64 {
    1.0
}

const fn default_scheduler_interval_seconds() -> f64 {
    5.0
}

const fn default_acting_token_weight() -> f64 {
    1.0
}

const fn default_acting_decay_tau_seconds() -> f64 {
    1.0
}

const fn default_buffer_per_program() -> usize {
    100
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AgentAwareConfig {
    pub pause_threshold: f64,
    pub pause_target: f64,
    pub resume_hysteresis: f64,
    pub resume_timeout_seconds: f64,
    pub resume_priority_boost: f64,
    pub scheduler_interval_seconds: f64,
    pub acting_token_weight: f64,
    pub acting_decay_tau_seconds: f64,
    pub buffer_per_program: usize,
}

impl Default for AgentAwareConfig {
    fn default() -> Self {
        Self {
            pause_threshold: default_pause_threshold(),
            pause_target: default_pause_target(),
            resume_hysteresis: default_resume_hysteresis(),
            resume_timeout_seconds: default_resume_timeout_seconds(),
            resume_priority_boost: default_resume_priority_boost(),
            scheduler_interval_seconds: default_scheduler_interval_seconds(),
            acting_token_weight: default_acting_token_weight(),
            acting_decay_tau_seconds: default_acting_decay_tau_seconds(),
            buffer_per_program: default_buffer_per_program(),
        }
    }
}

impl AgentAwareConfig {
    pub(super) fn validate(&self, location: &str) -> Result<(), RouterPolicyConfigError> {
        let valid_fraction = |value: f64| value.is_finite() && (0.0..=1.0).contains(&value);
        if !valid_fraction(self.pause_threshold) {
            return Err(invalid(
                location,
                "pause_threshold must be finite and in [0, 1]",
            ));
        }
        if !valid_fraction(self.pause_target) || self.pause_target > self.pause_threshold {
            return Err(invalid(
                location,
                "pause_target must be finite and in [0, pause_threshold]",
            ));
        }
        if !valid_fraction(self.resume_hysteresis) || self.resume_hysteresis > self.pause_threshold
        {
            return Err(invalid(
                location,
                "resume_hysteresis must be finite and in [0, pause_threshold]",
            ));
        }
        for (name, value) in [
            ("resume_timeout_seconds", self.resume_timeout_seconds),
            (
                "scheduler_interval_seconds",
                self.scheduler_interval_seconds,
            ),
            ("acting_token_weight", self.acting_token_weight),
            ("acting_decay_tau_seconds", self.acting_decay_tau_seconds),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(invalid(location, &format!("{name} must be finite and > 0")));
            }
        }
        if !self.resume_priority_boost.is_finite() || self.resume_priority_boost < 0.0 {
            return Err(invalid(
                location,
                "resume_priority_boost must be finite and >= 0",
            ));
        }
        Ok(())
    }

    pub fn scheduler_interval(&self) -> Duration {
        Duration::from_secs_f64(self.scheduler_interval_seconds)
    }
}

fn invalid(location: &str, message: &str) -> RouterPolicyConfigError {
    RouterPolicyConfigError::Validation(format!("{location} agent_aware {message}"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProgramStatus {
    Reasoning,
    Acting,
}

#[derive(Debug)]
struct Program {
    assigned_worker: Option<WorkerWithDpRank>,
    token_total: usize,
    step_count: usize,
    status: ProgramStatus,
    paused: bool,
    marked_for_pause: bool,
    acting_since: Option<Instant>,
    pending_since: Option<Instant>,
}

impl Program {
    fn new(token_total: usize, now: Instant) -> Self {
        Self {
            assigned_worker: None,
            token_total,
            step_count: 1,
            status: ProgramStatus::Reasoning,
            paused: false,
            marked_for_pause: false,
            acting_since: None,
            pending_since: Some(now),
        }
    }
}

pub(super) struct AgentAwarePolicy {
    config: AgentAwareConfig,
    programs: HashMap<String, Program>,
    requests: HashMap<String, String>,
    last_tick: Instant,
}

impl AgentAwarePolicy {
    pub(super) fn new(config: AgentAwareConfig) -> Self {
        Self {
            config,
            programs: HashMap::new(),
            requests: HashMap::new(),
            last_tick: Instant::now(),
        }
    }

    pub(super) fn prepare<C: WorkerConfigLike>(
        &mut self,
        request: &mut SchedulingRequest,
        workers: &HashMap<WorkerId, C>,
        block_size: u32,
        now: Instant,
    ) -> bool {
        if !request.mode.is_tracked() {
            return false;
        }
        let Some(session_id) = request.session_id.as_ref() else {
            return false;
        };

        let estimated_tokens = request.isl_tokens;
        let was_new = !self.programs.contains_key(session_id);
        let was_paused = self
            .programs
            .get(session_id)
            .is_some_and(|program| program.paused);
        if was_paused {
            request.priority_jump += self.config.resume_priority_boost;
        }
        let existing_assignment = self
            .programs
            .get(session_id)
            .and_then(|program| program.assigned_worker);
        let assignment = existing_assignment
            .filter(|worker| workers.contains_key(&worker.worker_id))
            .or(request.pinned_worker)
            .or_else(|| {
                was_new
                    .then(|| self.select_worker(request, workers, block_size, estimated_tokens))?
            });

        let has_capacity_metadata = self.has_capacity_metadata(request, workers);
        let program = self
            .programs
            .entry(session_id.clone())
            .or_insert_with(|| Program::new(estimated_tokens, now));
        if !was_new {
            program.step_count = program.step_count.saturating_add(1);
            program.token_total = estimated_tokens;
            program.status = ProgramStatus::Reasoning;
            program.acting_since = None;
        }
        program.assigned_worker = assignment;

        if was_new && assignment.is_none() && has_capacity_metadata {
            program.paused = true;
        }
        if program.paused {
            program.pending_since.get_or_insert(now);
            return true;
        }
        request.pinned_worker = program.assigned_worker;
        false
    }

    pub(super) fn apply_assignment(&self, request: &mut SchedulingRequest) {
        let Some(session_id) = request.session_id.as_ref() else {
            return;
        };
        if let Some(worker) = self
            .programs
            .get(session_id)
            .and_then(|program| (!program.paused).then_some(program.assigned_worker))
            .flatten()
        {
            request.pinned_worker = Some(worker);
        }
    }

    pub(super) fn can_dispatch(&self, request: &SchedulingRequest) -> bool {
        request
            .session_id
            .as_ref()
            .and_then(|session_id| self.programs.get(session_id))
            .is_none_or(|program| !program.paused)
    }

    pub(super) fn on_admitted(&mut self, request: &SchedulingRequest, worker: WorkerWithDpRank) {
        let (Some(request_id), Some(session_id)) = (
            request.mode.tracked_request_id(),
            request.session_id.as_ref(),
        ) else {
            return;
        };
        let Some(program) = self.programs.get_mut(session_id) else {
            return;
        };
        program.assigned_worker = Some(worker);
        program.paused = false;
        program.pending_since = None;
        program.status = ProgramStatus::Reasoning;
        self.requests
            .insert(request_id.to_string(), session_id.clone());
    }

    pub(super) fn on_admission_failed(&mut self, request: &SchedulingRequest) {
        let Some(session_id) = request.session_id.as_ref() else {
            return;
        };
        let remove = self
            .programs
            .get(session_id)
            .is_some_and(|program| program.step_count == 1);
        if remove {
            self.programs.remove(session_id);
        } else if let Some(program) = self.programs.get_mut(session_id) {
            program.step_count = program.step_count.saturating_sub(1);
            program.status = ProgramStatus::Acting;
            program.acting_since = Some(Instant::now());
        }
    }

    pub(super) fn complete(&mut self, request_id: &str, completion_tokens: usize, now: Instant) {
        let Some(session_id) = self.requests.remove(request_id) else {
            return;
        };
        let Some(program) = self.programs.get_mut(&session_id) else {
            return;
        };
        program.token_total = program.token_total.saturating_add(completion_tokens);
        program.status = ProgramStatus::Acting;
        program.acting_since = Some(now);
        if program.marked_for_pause {
            Self::pause(program);
        }
    }

    pub(super) fn end_session(&mut self, session_id: &str) {
        self.programs.remove(session_id);
        self.requests.retain(|_, value| value != session_id);
    }

    pub(super) fn tick<C: WorkerConfigLike>(
        &mut self,
        workers: &HashMap<WorkerId, C>,
        block_size: u32,
        now: Instant,
    ) {
        if now.duration_since(self.last_tick) < self.config.scheduler_interval() {
            return;
        }
        self.last_tick = now;
        let capacities = capacities(workers, block_size);
        if capacities.is_empty() {
            return;
        }
        self.force_expired(&capacities, now);
        self.greedy_resume(&capacities);
        self.pause_until_safe(&capacities);
    }

    fn has_capacity_metadata<C: WorkerConfigLike>(
        &self,
        request: &SchedulingRequest,
        workers: &HashMap<WorkerId, C>,
    ) -> bool {
        let eligibility = request.eligibility();
        eligibility
            .any_eligible_worker_rank(workers, |_, config| config.total_kv_blocks().is_some())
    }

    fn select_worker<C: WorkerConfigLike>(
        &self,
        request: &SchedulingRequest,
        workers: &HashMap<WorkerId, C>,
        block_size: u32,
        estimated_tokens: usize,
    ) -> Option<WorkerWithDpRank> {
        if self.programs.values().any(|program| program.paused) {
            return None;
        }
        let required = estimated_tokens.saturating_add(self.config.buffer_per_program);
        let eligibility = request.eligibility();
        let mut best = None;
        eligibility.for_each_eligible_worker_rank(workers, |worker, config| {
            let Some(capacity) = worker_capacity(config, block_size) else {
                return;
            };
            let used = self.worker_used(worker);
            if capacity.saturating_sub(used) < required {
                return;
            }
            if best.is_none_or(|(_, best_used)| used < best_used) {
                best = Some((worker, used));
            }
        });
        best.map(|(worker, _)| worker)
    }

    fn worker_used(&self, worker: WorkerWithDpRank) -> usize {
        self.programs
            .values()
            .filter(|program| !program.paused && program.assigned_worker == Some(worker))
            .fold(0usize, |used, program| {
                used.saturating_add(self.program_charge(program))
            })
    }

    fn program_charge(&self, program: &Program) -> usize {
        let tokens = match program.status {
            ProgramStatus::Reasoning => program.token_total,
            ProgramStatus::Acting => {
                (program.token_total as f64 * self.config.acting_token_weight) as usize
            }
        };
        tokens.saturating_add(self.config.buffer_per_program)
    }

    fn force_expired(&mut self, capacities: &HashMap<WorkerWithDpRank, usize>, now: Instant) {
        let timeout = Duration::from_secs_f64(self.config.resume_timeout_seconds);
        let expired: Vec<_> = self
            .programs
            .iter()
            .filter(|(_, program)| {
                program.paused
                    && program
                        .pending_since
                        .is_some_and(|since| now.duration_since(since) >= timeout)
            })
            .map(|(session_id, _)| session_id.clone())
            .collect();
        let mut used: HashMap<_, _> = capacities
            .keys()
            .map(|&worker| (worker, self.worker_used_decayed(worker, now)))
            .collect();
        for session_id in expired {
            let worker = capacities.keys().copied().max_by_key(|worker| {
                capacities[worker].saturating_sub(used.get(worker).copied().unwrap_or_default())
            });
            let charge = self
                .programs
                .get(&session_id)
                .map(|program| self.program_charge(program))
                .unwrap_or_default();
            if let Some(program) = self.programs.get_mut(&session_id) {
                program.assigned_worker = worker;
                program.paused = false;
                program.pending_since = None;
                if let Some(worker) = worker {
                    used.entry(worker)
                        .and_modify(|value| *value = value.saturating_add(charge));
                }
                tracing::warn!(session_id, "AgentAware forced session resume after timeout");
            }
        }
    }

    fn greedy_resume(&mut self, capacities: &HashMap<WorkerWithDpRank, usize>) {
        let ceiling = (self.config.pause_threshold - self.config.resume_hysteresis).max(0.0);
        let mut remaining: Vec<_> = capacities
            .iter()
            .map(|(&worker, &capacity)| {
                let ceiling = (capacity as f64 * ceiling) as usize;
                (worker, ceiling.saturating_sub(self.worker_used(worker)))
            })
            .filter(|(_, remaining)| *remaining > self.config.buffer_per_program)
            .collect();
        if remaining.is_empty() {
            return;
        }
        let mut paused: Vec<_> = self
            .programs
            .iter()
            .filter(|(_, program)| program.paused)
            .map(|(session_id, program)| {
                let group = if program.step_count > 1 && program.status == ProgramStatus::Reasoning
                {
                    0
                } else if program.step_count <= 1 {
                    1
                } else {
                    2
                };
                (
                    session_id.clone(),
                    group,
                    program.token_total,
                    program.pending_since,
                )
            })
            .collect();
        paused.sort_by_key(|(_, group, tokens, since)| (*group, *tokens, *since));
        let mut selected = Vec::new();
        let mut total = remaining.iter().map(|(_, value)| *value).sum::<usize>();
        for (session_id, _, tokens, _) in paused {
            let required = tokens.saturating_add(self.config.buffer_per_program);
            if required <= total {
                selected.push((session_id, tokens));
                total -= required;
            }
        }
        selected.sort_by_key(|(_, tokens)| std::cmp::Reverse(*tokens));
        remaining.sort_by_key(|(_, value)| std::cmp::Reverse(*value));
        let mut resumed = 0usize;
        for (session_id, tokens) in selected {
            let required = tokens.saturating_add(self.config.buffer_per_program);
            let fixed_worker = self.programs[&session_id].assigned_worker;
            let Some(index) = remaining.iter().position(|(worker, available)| {
                fixed_worker.is_none_or(|fixed| fixed == *worker) && *available >= required
            }) else {
                continue;
            };
            let (worker, available) = remaining[index];
            let program = self
                .programs
                .get_mut(&session_id)
                .expect("selected paused session must exist");
            program.assigned_worker = Some(worker);
            program.paused = false;
            program.pending_since = None;
            resumed += 1;
            remaining[index].1 = available - required;
            remaining.sort_by_key(|(_, value)| std::cmp::Reverse(*value));
        }
        if resumed > 0 {
            let still_paused = self
                .programs
                .values()
                .filter(|program| program.paused)
                .count();
            tracing::info!(resumed, still_paused, "AgentAware policy resumed sessions");
        }
    }

    fn pause_until_safe(&mut self, capacities: &HashMap<WorkerWithDpRank, usize>) {
        for (&worker, &capacity) in capacities {
            let threshold = (capacity as f64 * self.config.pause_threshold) as usize;
            let mut used = self.worker_used(worker);
            if used <= threshold {
                continue;
            }
            let base_used = used;
            let target = (capacity as f64 * self.config.pause_target) as usize;
            let mut candidates: Vec<_> = self
                .programs
                .iter()
                .filter(|(_, program)| {
                    !program.paused
                        && program.assigned_worker == Some(worker)
                        && !program.marked_for_pause
                })
                .map(|(session_id, program)| {
                    (
                        program.status != ProgramStatus::Acting,
                        program.token_total,
                        session_id.clone(),
                    )
                })
                .collect();
            candidates.sort_unstable();
            let mut paused = 0usize;
            let mut marked = 0usize;
            for (_, _, session_id) in candidates {
                if used <= target {
                    break;
                }
                let charge = self
                    .programs
                    .get(&session_id)
                    .map(|program| self.program_charge(program))
                    .unwrap_or_default();
                let program = self
                    .programs
                    .get_mut(&session_id)
                    .expect("pause candidate must exist");
                if program.status == ProgramStatus::Acting {
                    Self::pause(program);
                    used = used.saturating_sub(charge);
                    paused += 1;
                } else {
                    program.marked_for_pause = true;
                    marked += 1;
                }
            }
            if paused > 0 || marked > 0 {
                tracing::info!(
                    worker_id = worker.worker_id,
                    dp_rank = worker.dp_rank,
                    paused,
                    marked,
                    utilization_before = base_used as f64 / capacity as f64,
                    utilization_after = used as f64 / capacity as f64,
                    "AgentAware policy applied working-set pressure"
                );
            }
        }
    }

    fn pause(program: &mut Program) {
        program.paused = true;
        program.assigned_worker = None;
        program.marked_for_pause = false;
        program.pending_since = None;
    }

    fn worker_used_decayed(&self, worker: WorkerWithDpRank, now: Instant) -> usize {
        let tau = self.config.acting_decay_tau_seconds.max(1e-3);
        let mut count = 0usize;
        let tokens = self
            .programs
            .values()
            .filter(|program| !program.paused && program.assigned_worker == Some(worker))
            .map(|program| {
                count += 1;
                if program.status != ProgramStatus::Acting {
                    return program.token_total;
                }
                let idle = program
                    .acting_since
                    .map_or(0.0, |since| now.duration_since(since).as_secs_f64());
                (program.token_total as f64 * 2.0_f64.powf(-(idle / tau))) as usize
            })
            .sum::<usize>();
        tokens.saturating_add(count.saturating_mul(self.config.buffer_per_program))
    }
}

fn capacities<C: WorkerConfigLike>(
    workers: &HashMap<WorkerId, C>,
    block_size: u32,
) -> HashMap<WorkerWithDpRank, usize> {
    let mut capacities = HashMap::new();
    for (&worker_id, config) in workers {
        let Some(capacity) = worker_capacity(config, block_size) else {
            continue;
        };
        let start = config.data_parallel_start_rank();
        for rank in start..start.saturating_add(config.data_parallel_size()) {
            capacities.insert(WorkerWithDpRank::new(worker_id, rank), capacity);
        }
    }
    capacities
}

fn worker_capacity(config: &impl WorkerConfigLike, block_size: u32) -> Option<usize> {
    let device = usize::try_from(config.total_kv_blocks()?)
        .ok()?
        .checked_mul(block_size as usize)?;
    let extra = usize::try_from(config.extra_kv_capacity_tokens().unwrap_or_default()).ok()?;
    device.checked_add(extra)
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashMap;

    use super::*;
    use crate::protocols::RoutingConstraints;
    use crate::scheduling::{OverlapSignals, ScheduleMode};
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
        for (request_id, session_id, tokens) in [("big-r1", "big", 600), ("small-r1", "small", 200)]
        {
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
}

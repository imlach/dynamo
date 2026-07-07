// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use tokio::time::Instant;

use super::super::types::SchedulingRequest;
use super::control::worker_capacity;
use super::{AgentAwareConfig, AgentAwarePolicy, Program, ProgramStatus};
use crate::protocols::{WorkerConfigLike, WorkerId, WorkerWithDpRank};

impl AgentAwarePolicy {
    pub(in crate::scheduling) fn new(config: AgentAwareConfig) -> Self {
        Self {
            config,
            programs: HashMap::new(),
            requests: HashMap::new(),
            last_tick: Instant::now(),
        }
    }

    pub(in crate::scheduling) fn prepare<C: WorkerConfigLike>(
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

    pub(in crate::scheduling) fn apply_assignment(&self, request: &mut SchedulingRequest) {
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

    pub(in crate::scheduling) fn can_dispatch(&self, request: &SchedulingRequest) -> bool {
        request
            .session_id
            .as_ref()
            .and_then(|session_id| self.programs.get(session_id))
            .is_none_or(|program| !program.paused)
    }

    pub(in crate::scheduling) fn on_admitted(
        &mut self,
        request: &SchedulingRequest,
        worker: WorkerWithDpRank,
    ) {
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

    pub(in crate::scheduling) fn on_admission_failed(&mut self, request: &SchedulingRequest) {
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

    pub(in crate::scheduling) fn complete(
        &mut self,
        request_id: &str,
        completion_tokens: usize,
        now: Instant,
    ) {
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

    pub(in crate::scheduling) fn end_session(&mut self, session_id: &str) {
        self.programs.remove(session_id);
        self.requests.retain(|_, value| value != session_id);
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

    pub(super) fn worker_used(&self, worker: WorkerWithDpRank) -> usize {
        self.programs
            .values()
            .filter(|program| !program.paused && program.assigned_worker == Some(worker))
            .fold(0usize, |used, program| {
                used.saturating_add(self.program_charge(program))
            })
    }

    pub(super) fn program_charge(&self, program: &Program) -> usize {
        let tokens = match program.status {
            ProgramStatus::Reasoning => program.token_total,
            ProgramStatus::Acting => {
                (program.token_total as f64 * self.config.acting_token_weight) as usize
            }
        };
        tokens.saturating_add(self.config.buffer_per_program)
    }
}

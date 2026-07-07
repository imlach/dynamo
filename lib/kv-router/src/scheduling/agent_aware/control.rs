// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use tokio::time::{Duration, Instant};

use super::{AgentAwarePolicy, Program, ProgramStatus};
use crate::protocols::{WorkerConfigLike, WorkerId, WorkerWithDpRank};

impl AgentAwarePolicy {
    pub(in crate::scheduling) fn tick<C: WorkerConfigLike>(
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

    pub(super) fn pause(program: &mut Program) {
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

pub(super) fn worker_capacity(config: &impl WorkerConfigLike, block_size: u32) -> Option<usize> {
    let device = usize::try_from(config.total_kv_blocks()?)
        .ok()?
        .checked_mul(block_size as usize)?;
    let extra = usize::try_from(config.extra_kv_capacity_tokens().unwrap_or_default()).ok()?;
    device.checked_add(extra)
}

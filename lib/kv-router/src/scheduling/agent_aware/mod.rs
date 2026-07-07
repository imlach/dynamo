// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use tokio::time::Instant;

use crate::protocols::WorkerWithDpRank;

mod config;
mod control;
mod lifecycle;

pub use config::AgentAwareConfig;

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

pub(in crate::scheduling) struct AgentAwarePolicy {
    config: AgentAwareConfig,
    programs: HashMap<String, Program>,
    requests: HashMap<String, String>,
    last_tick: Instant,
}

#[cfg(test)]
mod tests;

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::Deserialize;
use tokio::time::Duration;

use super::super::policy_config::RouterPolicyConfigError;

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
    pub(in crate::scheduling) fn validate(
        &self,
        location: &str,
    ) -> Result<(), RouterPolicyConfigError> {
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

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in validation rules, grouped by the contract that owns them.
//!
//! These helpers inspect already parsed values. They do not run during Serde
//! deserialization or builder construction, insert defaults, or mutate requests.
//! Callers choose the rules and their order, retain cross-field and contextual
//! checks, and map failures to their transport's error representation.

/// A validation failure whose display text is the rule's diagnostic, without a prefix.
///
/// This error carries no HTTP status or transport category. Dynamo callers that
/// preserve their existing untyped error contract should copy its display text
/// into their local error representation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ValidationError(String);

/// Scalar admission rules ported from Dynamo's chat and completion validators.
///
/// These are Dynamo's rules, not a claim that OpenAI or any inference backend
/// accepts exactly the same values. Each helper accepts `None`, leaves the
/// supplied value unchanged, and returns a [`ValidationError`] on rejection.
/// Parsing determines whether an omitted field or JSON `null` becomes `None`.
/// Integer representation limits apply before validation.
///
/// For example, a caller can check a scalar while retaining responsibility for
/// the request's other fields and cross-field constraints:
///
/// ```
/// use dynamo_protocols::validation::dynamo::validate_top_p;
///
/// assert!(validate_top_p(None).is_ok());
/// assert!(validate_top_p(Some(0.5)).is_ok());
/// assert_eq!(
///     validate_top_p(Some(0.0)).unwrap_err().to_string(),
///     "Top_p must be between 0 and 1, got 0",
/// );
/// ```
pub mod dynamo {
    use super::ValidationError;

    /// Inclusive lower bound for temperature.
    pub const MIN_TEMPERATURE: f32 = 0.0;
    /// Inclusive upper bound for temperature.
    pub const MAX_TEMPERATURE: f32 = 2.0;
    /// Exclusive lower bound for top_p.
    pub const MIN_TOP_P: f32 = 0.0;
    /// Inclusive upper bound for top_p.
    pub const MAX_TOP_P: f32 = 1.0;
    /// Inclusive lower bound for min_p.
    pub const MIN_MIN_P: f32 = 0.0;
    /// Inclusive upper bound for min_p.
    pub const MAX_MIN_P: f32 = 1.0;
    /// Inclusive lower bound for frequency penalty.
    pub const MIN_FREQUENCY_PENALTY: f32 = -2.0;
    /// Inclusive upper bound for frequency penalty.
    pub const MAX_FREQUENCY_PENALTY: f32 = 2.0;
    /// Inclusive lower bound for presence penalty.
    pub const MIN_PRESENCE_PENALTY: f32 = -2.0;
    /// Inclusive upper bound for presence penalty.
    pub const MAX_PRESENCE_PENALTY: f32 = 2.0;
    /// Inclusive lower bound for chat top_logprobs.
    pub const MIN_TOP_LOGPROBS: u8 = 0;
    /// Inclusive upper bound for chat top_logprobs.
    pub const MAX_TOP_LOGPROBS: u8 = 20;
    /// Inclusive lower bound for completion logprobs.
    pub const MIN_LOGPROBS: u8 = 0;
    /// Inclusive upper bound for completion logprobs under Dynamo's contract.
    pub const MAX_LOGPROBS: u8 = 20;
    /// Inclusive lower bound for the number of choices.
    pub const MIN_N: u8 = 1;
    /// Inclusive upper bound for the number of choices.
    pub const MAX_N: u8 = 128;
    /// Exclusive lower bound for finite repetition penalty values.
    pub const MIN_REPETITION_PENALTY: f32 = 0.0;
    /// Inclusive upper bound for finite repetition penalty values.
    pub const MAX_REPETITION_PENALTY: f32 = 2.0;

    const MAX_OUTPUT_TOKENS: u32 = 1_048_576;

    /// Checks Dynamo's temperature range: `None` or a finite value in `[0, 2]`.
    /// Returns an error for other values; does not mutate or default the input.
    pub fn validate_temperature(temperature: Option<f32>) -> Result<(), ValidationError> {
        if let Some(temp) = temperature
            && !(MIN_TEMPERATURE..=MAX_TEMPERATURE).contains(&temp)
        {
            return Err(ValidationError(format!(
                "Temperature must be between {} and {}, got {}",
                MIN_TEMPERATURE, MAX_TEMPERATURE, temp
            )));
        }
        Ok(())
    }

    /// Checks Dynamo's top_p range: `None` or a finite value in `(0, 1]`.
    /// Returns an error for other values, including zero; does not mutate or default the input.
    pub fn validate_top_p(top_p: Option<f32>) -> Result<(), ValidationError> {
        if let Some(p) = top_p
            && !(p.is_finite() && p > MIN_TOP_P && p <= MAX_TOP_P)
        {
            return Err(ValidationError(format!(
                "Top_p must be between {} and {}, got {}",
                MIN_TOP_P, MAX_TOP_P, p
            )));
        }
        Ok(())
    }

    /// Accepts `None` or top_k values at least `-1`, including both `-1` and `0`.
    /// Returns an error below `-1`; does not interpret sentinels or mutate the input.
    pub fn validate_top_k(top_k: Option<i32>) -> Result<(), ValidationError> {
        match top_k {
            None => Ok(()),
            Some(k) if k >= -1 => Ok(()),
            _ => Err(ValidationError(
                "Top_k must be null or greater than or equal to -1".to_string(),
            )),
        }
    }

    /// Checks frequency penalty: `None` or a finite value in `[-2, 2]`.
    /// Returns an error for other values; does not mutate or default the input.
    pub fn validate_frequency_penalty(
        frequency_penalty: Option<f32>,
    ) -> Result<(), ValidationError> {
        if let Some(penalty) = frequency_penalty
            && !(MIN_FREQUENCY_PENALTY..=MAX_FREQUENCY_PENALTY).contains(&penalty)
        {
            return Err(ValidationError(format!(
                "Frequency penalty must be between {} and {}, got {}",
                MIN_FREQUENCY_PENALTY, MAX_FREQUENCY_PENALTY, penalty
            )));
        }
        Ok(())
    }

    /// Checks presence penalty: `None` or a finite value in `[-2, 2]`.
    /// Returns an error for other values; does not mutate or default the input.
    pub fn validate_presence_penalty(presence_penalty: Option<f32>) -> Result<(), ValidationError> {
        if let Some(penalty) = presence_penalty
            && !(MIN_PRESENCE_PENALTY..=MAX_PRESENCE_PENALTY).contains(&penalty)
        {
            return Err(ValidationError(format!(
                "Presence penalty must be between {} and {}, got {}",
                MIN_PRESENCE_PENALTY, MAX_PRESENCE_PENALTY, penalty
            )));
        }
        Ok(())
    }

    /// Preserves Dynamo's repetition-penalty check, accepting `None` or values in `(0, 2]`.
    ///
    /// The existing comparison also accepts a typed `f32::NAN`; JSON cannot
    /// represent that value. Zero, negative values, values above 2, and infinities
    /// return an error. This helper does not mutate or default the input.
    pub fn validate_repetition_penalty(
        repetition_penalty: Option<f32>,
    ) -> Result<(), ValidationError> {
        if let Some(penalty) = repetition_penalty
            && (penalty <= MIN_REPETITION_PENALTY || penalty > MAX_REPETITION_PENALTY)
        {
            return Err(ValidationError(format!(
                "Repetition penalty must be between {} and {}, got {}",
                MIN_REPETITION_PENALTY, MAX_REPETITION_PENALTY, penalty
            )));
        }
        Ok(())
    }

    /// Checks min_p: `None` or a finite value in `[0, 1]`.
    /// Returns an error for other values; does not mutate or default the input.
    pub fn validate_min_p(min_p: Option<f32>) -> Result<(), ValidationError> {
        if let Some(p) = min_p
            && !(MIN_MIN_P..=MAX_MIN_P).contains(&p)
        {
            return Err(ValidationError(format!(
                "Min_p must be between {} and {}, got {}",
                MIN_MIN_P, MAX_MIN_P, p
            )));
        }
        Ok(())
    }

    /// Checks the number of choices: `None` or a value in `[1, 128]`.
    /// Returns an error otherwise; does not insert a default or check other fields.
    pub fn validate_n(n: Option<u8>) -> Result<(), ValidationError> {
        if let Some(value) = n
            && !(MIN_N..=MAX_N).contains(&value)
        {
            return Err(ValidationError(format!(
                "n must be between {} and {}, got {}",
                MIN_N, MAX_N, value
            )));
        }
        Ok(())
    }

    /// Checks chat top_logprobs: `None` or a value in `[0, 20]`.
    /// Returns an error above 20; does not mutate the input or check the logprobs flag.
    pub fn validate_top_logprobs(top_logprobs: Option<u8>) -> Result<(), ValidationError> {
        if let Some(value) = top_logprobs
            && !(0..=20).contains(&value)
        {
            return Err(ValidationError(format!(
                "Top_logprobs must be between 0 and {}, got {}",
                MAX_TOP_LOGPROBS, value
            )));
        }
        Ok(())
    }

    /// Checks completion logprobs: `None` or a value in `[0, 20]`.
    /// Returns an error above Dynamo's cap of 20; does not mutate or default the input.
    pub fn validate_logprobs(logprobs: Option<u8>) -> Result<(), ValidationError> {
        if let Some(value) = logprobs
            && !(MIN_LOGPROBS..=MAX_LOGPROBS).contains(&value)
        {
            return Err(ValidationError(format!(
                "Logprobs must be between 0 and {}, got {}",
                MAX_LOGPROBS, value
            )));
        }
        Ok(())
    }

    /// Checks max_tokens: `None` or a value in `[1, 1_048_576]`.
    /// Returns a field-specific error outside that range; does not mutate or default the input.
    pub fn validate_max_tokens(max_tokens: Option<u32>) -> Result<(), ValidationError> {
        if let Some(tokens) = max_tokens
            && tokens == 0
        {
            return Err(ValidationError(format!(
                "Max tokens must be greater than 0, got {}",
                tokens
            )));
        }
        if let Some(tokens) = max_tokens
            && tokens > MAX_OUTPUT_TOKENS
        {
            return Err(ValidationError(format!(
                "Max tokens must not exceed {}, got {}",
                MAX_OUTPUT_TOKENS, tokens
            )));
        }
        Ok(())
    }

    /// Checks max_completion_tokens: `None` or a value in `[1, 1_048_576]`.
    /// Returns a field-specific error outside that range; does not mutate or default the input.
    pub fn validate_max_completion_tokens(
        max_completion_tokens: Option<u32>,
    ) -> Result<(), ValidationError> {
        if let Some(tokens) = max_completion_tokens
            && tokens == 0
        {
            return Err(ValidationError(format!(
                "Max completion tokens must be greater than 0, got {}",
                tokens
            )));
        }
        if let Some(tokens) = max_completion_tokens
            && tokens > MAX_OUTPUT_TOKENS
        {
            return Err(ValidationError(format!(
                "Max completion tokens must not exceed {}, got {}",
                MAX_OUTPUT_TOKENS, tokens
            )));
        }
        Ok(())
    }
}

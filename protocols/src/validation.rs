// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in, partial validation of parsed protocol values.
//!
//! Helpers and request traits never mutate, normalize, or dispatch a request.
//! Success means only that the documented checks passed, not complete admission.
#![doc = include_str!("validation/coverage.md")]

/// A library diagnostic for the first failed covered check.
///
/// This type has no HTTP status or backend exception metadata. Consumers preserving
/// an existing server contract must adapt failures to their own diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{field}: {message}")]
pub struct ValidationError {
    /// The request field whose covered check failed.
    pub field: &'static str,
    /// The library constraint description, without a transport prefix.
    pub message: &'static str,
    /// Rust Display of the rejected explicit number, formatted only on failure.
    /// This preserves typed nonfinite values such as NaN/inf; it is not the original JSON text.
    pub value: String,
}

/// Accepts None or a value satisfying the predicate; formats rejected values only on failure.
/// Returns the supplied library diagnostic without mutating or defaulting the input.
fn check<T: Copy + std::fmt::Display>(
    value: Option<T>,
    accepted: impl FnOnce(T) -> bool,
    field: &'static str,
    message: &'static str,
) -> Result<(), ValidationError> {
    match value {
        Some(value) if !accepted(value) => Err(ValidationError {
            field,
            message,
            value: value.to_string(),
        }),
        _ => Ok(()),
    }
}

/// Documented OpenAI numeric rules at the parsed-value boundary.
pub mod openai {
    use super::{ValidationError, check};
    use crate::types::{CreateChatCompletionRequest, CreateCompletionRequest};

    /// Returns whether a parsed temperature is finite and in [0, 2]; rejects NaN and infinities.
    pub fn is_valid_temperature(value: f32) -> bool {
        (0.0..=2.0).contains(&value)
    }

    /// Accepts None or temperature that is finite and in [0, 2], without mutation or defaults.
    /// Other explicit values return a library diagnostic, not a backend HTTP error.
    pub fn validate_temperature(value: Option<f32>) -> Result<(), ValidationError> {
        check(
            value,
            is_valid_temperature,
            "temperature",
            "must be finite and in [0, 2]",
        )
    }

    /// Returns whether a parsed top_p is finite and in [0, 1]; rejects NaN and infinities.
    pub fn is_valid_top_p(value: f32) -> bool {
        (0.0..=1.0).contains(&value)
    }

    /// Accepts None or top_p that is finite and in [0, 1], without mutation or defaults.
    /// Other explicit values return a library diagnostic, not a backend HTTP error.
    pub fn validate_top_p(value: Option<f32>) -> Result<(), ValidationError> {
        check(
            value,
            is_valid_top_p,
            "top_p",
            "must be finite and in [0, 1]",
        )
    }

    /// Returns whether a parsed frequency_penalty is finite and in [-2, 2]; rejects NaN and infinities.
    pub fn is_valid_frequency_penalty(value: f32) -> bool {
        (-2.0..=2.0).contains(&value)
    }

    /// Accepts None or frequency_penalty that is finite and in [-2, 2], without mutation or defaults.
    /// Other explicit values return a library diagnostic, not a backend HTTP error.
    pub fn validate_frequency_penalty(value: Option<f32>) -> Result<(), ValidationError> {
        check(
            value,
            is_valid_frequency_penalty,
            "frequency_penalty",
            "must be finite and in [-2, 2]",
        )
    }

    /// Returns whether a parsed presence_penalty is finite and in [-2, 2]; rejects NaN and infinities.
    pub fn is_valid_presence_penalty(value: f32) -> bool {
        (-2.0..=2.0).contains(&value)
    }

    /// Accepts None or presence_penalty that is finite and in [-2, 2], without mutation or defaults.
    /// Other explicit values return a library diagnostic, not a backend HTTP error.
    pub fn validate_presence_penalty(value: Option<f32>) -> Result<(), ValidationError> {
        check(
            value,
            is_valid_presence_penalty,
            "presence_penalty",
            "must be finite and in [-2, 2]",
        )
    }

    /// Accepts unspecified completion logprobs or 0..=5 without mutation or defaults.
    ///
    /// Negative values, including -1, and values above 5 return a logprobs diagnostic.
    /// This is the integer /v1/completions field, not chat's boolean logprobs.
    /// Pass the existing u8 request field with `value.map(i32::from)`.
    pub fn validate_completion_logprobs(value: Option<i32>) -> Result<(), ValidationError> {
        check(
            value,
            |value| (0..=5).contains(&value),
            "logprobs",
            "must be in [0, 5]",
        )
    }

    /// Accepts unspecified chat top_logprobs or 0..=20, without mutation or defaults.
    /// Other counts return a top_logprobs diagnostic. The relationship to chat's
    /// boolean logprobs field is a separate, uncovered operation.
    pub fn validate_top_logprobs(value: Option<u8>) -> Result<(), ValidationError> {
        check(
            value,
            |value| value <= 20,
            "top_logprobs",
            "must be in [0, 20]",
        )
    }

    /// Accepts unspecified n or 1..=128, without mutation or default insertion.
    /// Other counts return an n diagnostic; batch totals and sampling interactions are separate.
    pub fn validate_n(value: Option<u8>) -> Result<(), ValidationError> {
        check(
            value,
            |value| (1..=128).contains(&value),
            "n",
            "must be in [1, 128]",
        )
    }

    /// Accepts unspecified completion best_of or 0..=20, without mutation or defaults.
    /// Other counts return a best_of diagnostic. Relationships to n and streaming
    /// are separate, uncovered operations; this helper checks only the count.
    pub fn validate_best_of(value: Option<u8>) -> Result<(), ValidationError> {
        check(value, |value| value <= 20, "best_of", "must be in [0, 20]")
    }

    /// Checks a parsed f32 logit-bias value against the finite range [-100, 100].
    /// Returns a logit_bias diagnostic for other values, without mutation or clamping.
    ///
    /// The caller owns JSON numeric admission, precision conversion, token-ID policy,
    /// and map traversal order. This does not validate a complete logit_bias map.
    /// The returned value uses Rust Display of this f32, not an earlier f64 or JSON lexeme.
    pub fn validate_logit_bias_value(value: f32) -> Result<(), ValidationError> {
        check(
            Some(value),
            |value| (-100.0..=100.0).contains(&value),
            "logit_bias",
            "must be finite and in [-100, 100]",
        )
    }

    /// Partial, read-only validation for the two owned completion request types.
    ///
    /// Order: temperature, top_p, frequency_penalty, presence_penalty, then
    /// completion logprobs (legacy completion only), then n, then chat top_logprobs
    /// or completion best_of. All other fields and cross-field
    /// checks are uncovered. None passes at this parsed boundary; engine JSON-null
    /// admission and model/runtime constraints are not covered. See module coverage.
    pub trait OpenAIValidation {
        /// Returns the first covered failure in the documented order without mutation.
        /// Success means only that the covered checks passed, not full endpoint admission.
        fn validate(&self) -> Result<(), ValidationError>;
    }

    impl OpenAIValidation for CreateChatCompletionRequest {
        /// Checks this request's covered fields in the trait's order without normalization.
        fn validate(&self) -> Result<(), ValidationError> {
            validate_temperature(self.temperature)?;
            validate_top_p(self.top_p)?;
            validate_frequency_penalty(self.frequency_penalty)?;
            validate_presence_penalty(self.presence_penalty)?;
            validate_n(self.n)?;
            validate_top_logprobs(self.top_logprobs)
        }
    }

    impl OpenAIValidation for CreateCompletionRequest {
        /// Checks this request's covered fields in the trait's order without normalization.
        fn validate(&self) -> Result<(), ValidationError> {
            validate_temperature(self.temperature)?;
            validate_top_p(self.top_p)?;
            validate_frequency_penalty(self.frequency_penalty)?;
            validate_presence_penalty(self.presence_penalty)?;
            validate_completion_logprobs(self.logprobs.map(i32::from))?;
            validate_n(self.n)?;
            validate_best_of(self.best_of)
        }
    }
}

/// Common engine predicates outside the documented OpenAI baseline.
///
/// Only numeric predicates are shared. Null admission, defaults, diagnostics,
/// check order, and later greedy transformations are not covered by these functions.
pub mod shared {
    /// Returns whether parsed top_p is in (0, 1]; rejects zero and nonfinite values.
    /// Both pinned engines use this numeric predicate; OpenAI's baseline includes zero.
    pub fn is_valid_top_p(value: f32) -> bool {
        value > 0.0 && value <= 1.0
    }

    /// Returns whether parsed min_p is in [0, 1]; rejects NaN and infinities.
    /// Both pinned engines check this extension after their own default resolution.
    /// The owned request types have no min_p field, so request traits do not call it.
    pub fn is_valid_min_p(value: f32) -> bool {
        (0.0..=1.0).contains(&value)
    }
}

/// Partial vLLM-oriented library contract, including a deliberate fixed logprobs policy.
pub mod vllm {
    use super::{ValidationError, check};
    use crate::types::{CreateChatCompletionRequest, CreateCompletionRequest};

    pub use super::openai::{
        is_valid_frequency_penalty, is_valid_presence_penalty, is_valid_temperature,
        validate_frequency_penalty, validate_presence_penalty, validate_temperature,
    };
    pub use super::shared::{is_valid_min_p, is_valid_top_p};

    /// Accepts None or finite min_p in [0, 1], without mutation or defaults.
    /// Other explicit values return a library diagnostic carrying their Rust Display.
    /// This adapter does not establish engine HTTP-null admission or backend diagnostics.
    pub fn validate_min_p(value: Option<f32>) -> Result<(), ValidationError> {
        check(
            value,
            is_valid_min_p,
            "min_p",
            "must be finite and in [0, 1]",
        )
    }

    /// Accepts None or top_p that is finite and in (0, 1], without mutation or defaults.
    /// Other explicit values return a library diagnostic, not a backend HTTP error.
    pub fn validate_top_p(value: Option<f32>) -> Result<(), ValidationError> {
        check(
            value,
            is_valid_top_p,
            "top_p",
            "must be finite and in (0, 1]",
        )
    }

    /// Accepts unspecified completion logprobs or 0..=20 without mutation or defaults.
    ///
    /// Negative values, including -1, and values above 20 return a logprobs diagnostic.
    /// This is the integer /v1/completions field, not chat's boolean logprobs.
    /// Pass the existing u8 request field with `value.map(i32::from)`.
    ///
    /// DEP #16992 selects this fixed library policy without model context. Actual
    /// vLLM also supports a full-vocabulary sentinel and configurable limits, so
    /// this helper deliberately does not reproduce complete vLLM admission.
    pub fn validate_completion_logprobs(value: Option<i32>) -> Result<(), ValidationError> {
        check(
            value,
            |value| (0..=20).contains(&value),
            "logprobs",
            "must be in [0, 20]",
        )
    }

    /// Partial, read-only validation for the two owned completion request types.
    ///
    /// Order: temperature, top_p, frequency_penalty, presence_penalty, then
    /// completion logprobs (legacy completion only). All other fields and cross-field
    /// checks are uncovered. None passes at this parsed boundary; engine JSON-null
    /// admission and model/runtime constraints are not covered. See module coverage.
    pub trait VllmValidation {
        /// Returns the first covered failure in the documented order without mutation.
        /// Success means only that the covered checks passed, not full endpoint admission.
        fn validate(&self) -> Result<(), ValidationError>;
    }

    impl VllmValidation for CreateChatCompletionRequest {
        /// Checks this request's covered fields in the trait's order without normalization.
        fn validate(&self) -> Result<(), ValidationError> {
            validate_temperature(self.temperature)?;
            validate_top_p(self.top_p)?;
            validate_frequency_penalty(self.frequency_penalty)?;
            validate_presence_penalty(self.presence_penalty)?;
            Ok(())
        }
    }

    impl VllmValidation for CreateCompletionRequest {
        /// Checks this request's covered fields in the trait's order without normalization.
        fn validate(&self) -> Result<(), ValidationError> {
            validate_temperature(self.temperature)?;
            validate_top_p(self.top_p)?;
            validate_frequency_penalty(self.frequency_penalty)?;
            validate_presence_penalty(self.presence_penalty)?;
            validate_completion_logprobs(self.logprobs.map(i32::from))
        }
    }
}

/// Partial SGLang numeric contract for parsed request values.
pub mod sglang {
    use super::{ValidationError, check};
    use crate::types::{CreateChatCompletionRequest, CreateCompletionRequest};

    pub use super::openai::{
        is_valid_frequency_penalty, is_valid_presence_penalty, validate_frequency_penalty,
        validate_presence_penalty,
    };
    pub use super::shared::{is_valid_min_p, is_valid_top_p};

    /// Accepts None or finite min_p in [0, 1], without mutation or defaults.
    /// Other explicit values return a library diagnostic carrying their Rust Display.
    /// This adapter does not establish engine HTTP-null admission or backend diagnostics.
    pub fn validate_min_p(value: Option<f32>) -> Result<(), ValidationError> {
        check(
            value,
            is_valid_min_p,
            "min_p",
            "must be finite and in [0, 1]",
        )
    }

    /// Returns whether a parsed temperature is finite and nonnegative; rejects NaN and infinities.
    pub fn is_valid_temperature(value: f32) -> bool {
        value.is_finite() && value >= 0.0
    }

    /// Accepts None or temperature that is finite and nonnegative, without mutation or defaults.
    /// Other explicit values return a library diagnostic, not a backend HTTP error.
    pub fn validate_temperature(value: Option<f32>) -> Result<(), ValidationError> {
        check(
            value,
            is_valid_temperature,
            "temperature",
            "must be finite and nonnegative",
        )
    }

    /// Accepts None or top_p that is finite and in (0, 1], without mutation or defaults.
    /// Other explicit values return a library diagnostic, not a backend HTTP error.
    pub fn validate_top_p(value: Option<f32>) -> Result<(), ValidationError> {
        check(
            value,
            is_valid_top_p,
            "top_p",
            "must be finite and in (0, 1]",
        )
    }

    /// Partial, read-only validation for the two owned completion request types.
    ///
    /// Order: temperature, top_p, frequency_penalty, presence_penalty. All other fields and cross-field
    /// checks are uncovered. None passes at this parsed boundary; engine JSON-null
    /// admission and model/runtime constraints are not covered. See module coverage.
    pub trait SglangValidation {
        /// Returns the first covered failure in the documented order without mutation.
        /// Success means only that the covered checks passed, not full endpoint admission.
        fn validate(&self) -> Result<(), ValidationError>;
    }

    impl SglangValidation for CreateChatCompletionRequest {
        /// Checks this request's covered fields in the trait's order without normalization.
        fn validate(&self) -> Result<(), ValidationError> {
            validate_temperature(self.temperature)?;
            validate_top_p(self.top_p)?;
            validate_frequency_penalty(self.frequency_penalty)?;
            validate_presence_penalty(self.presence_penalty)?;
            Ok(())
        }
    }

    impl SglangValidation for CreateCompletionRequest {
        /// Checks this request's covered fields in the trait's order without normalization.
        fn validate(&self) -> Result<(), ValidationError> {
            validate_temperature(self.temperature)?;
            validate_top_p(self.top_p)?;
            validate_frequency_penalty(self.frequency_penalty)?;
            validate_presence_penalty(self.presence_penalty)?;
            Ok(())
        }
    }
}

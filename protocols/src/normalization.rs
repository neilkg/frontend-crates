// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit normalization at the parsed request boundary, separate from validation.
//!
//! Only preservation of legacy completion logprobs is covered. Model defaults,
//! sentinel rewrites, greedy transformations, and request-wide normalization are not covered.

/// OpenAI baseline preservation at the owned completion request's typed boundary.
pub mod openai {
    /// Preserves completion logprobs unchanged, including unspecified and invalid values.
    ///
    /// Call after parsing and before the selected logprobs validator. The owned
    /// `CreateCompletionRequest` already maps missing and explicit null to None;
    /// serialization skips the key for None. This operation cannot recover that
    /// distinction and neither changes parsing nor supplies an effective default.
    ///
    /// All Some values, including zero and out-of-range counts, remain available
    /// for validation. No context is required; this operation cannot fail, has no
    /// side effects, and is idempotent. It does not imply that validation passed.
    /// Later engine conversion owns effective logprob enablement/count selection.
    /// See [`crate::validation`] for source-qualified coverage and a usage example.
    pub fn normalize_completion_logprobs(_value: &mut Option<u8>) {}
}

/// vLLM reuse of baseline preservation, before engine conversion.
pub mod vllm {
    pub use super::openai::normalize_completion_logprobs;
}

/// SGLang reuse of baseline preservation, before engine conversion.
pub mod sglang {
    pub use super::openai::normalize_completion_logprobs;
}

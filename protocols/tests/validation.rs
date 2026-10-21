// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_protocols::{
    types::{CreateChatCompletionRequest, CreateCompletionRequest},
    validation::{
        ValidationError,
        openai::{self, OpenAIValidation},
        sglang::{self, SglangValidation},
        shared,
        vllm::{self, VllmValidation},
    },
};

type ChatValidator = fn(&CreateChatCompletionRequest) -> Result<(), ValidationError>;
type CompletionValidator = fn(&CreateCompletionRequest) -> Result<(), ValidationError>;

const CHAT_VALIDATORS: [ChatValidator; 3] = [
    OpenAIValidation::validate,
    VllmValidation::validate,
    SglangValidation::validate,
];
const COMPLETION_VALIDATORS: [CompletionValidator; 3] = [
    OpenAIValidation::validate,
    VllmValidation::validate,
    SglangValidation::validate,
];

#[test]
fn baseline_and_engine_predicates_cover_bounds_and_nonfinite_values() {
    for predicate in [
        openai::is_valid_temperature as fn(f32) -> bool,
        vllm::is_valid_temperature,
    ] {
        for value in [0.0, -0.0, 1.0, 2.0] {
            assert!(predicate(value));
        }
        for value in [-0.01, 2.01, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(!predicate(value));
        }
    }
    assert!(sglang::is_valid_temperature(3.0));
    assert!(sglang::is_valid_temperature(f32::MAX));
    for value in [-0.01, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(!sglang::is_valid_temperature(value));
    }
    for predicate in [
        openai::is_valid_frequency_penalty as fn(f32) -> bool,
        openai::is_valid_presence_penalty,
        vllm::is_valid_frequency_penalty,
        vllm::is_valid_presence_penalty,
        sglang::is_valid_frequency_penalty,
        sglang::is_valid_presence_penalty,
    ] {
        for value in [-2.0, 0.0, 2.0] {
            assert!(predicate(value));
        }
        for value in [-2.01, 2.01, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(!predicate(value));
        }
    }
}

#[test]
fn shared_extensions_are_exercised_through_both_engines() {
    assert!(openai::is_valid_top_p(0.0));
    for predicate in [
        shared::is_valid_top_p as fn(f32) -> bool,
        vllm::is_valid_top_p,
        sglang::is_valid_top_p,
    ] {
        for value in [f32::MIN_POSITIVE, 0.5, 1.0] {
            assert!(predicate(value));
        }
        for value in [
            0.0,
            -0.0,
            -0.1,
            1.01,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            assert!(!predicate(value));
        }
    }
    for predicate in [
        shared::is_valid_min_p as fn(f32) -> bool,
        vllm::is_valid_min_p,
        sglang::is_valid_min_p,
        openai::is_valid_top_p,
    ] {
        for value in [0.0, 0.5, 1.0] {
            assert!(predicate(value));
        }
        for value in [-0.1, 1.01, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(!predicate(value));
        }
    }
}

#[test]
fn fixed_completion_logprobs_contract_has_distinguishing_counts() {
    for value in [None, Some(0), Some(5)] {
        assert!(openai::validate_completion_logprobs(value).is_ok());
        assert!(vllm::validate_completion_logprobs(value).is_ok());
    }
    for value in [Some(6), Some(10), Some(20)] {
        assert!(openai::validate_completion_logprobs(value).is_err());
        assert!(vllm::validate_completion_logprobs(value).is_ok());
    }
    for value in [Some(i32::MIN), Some(-1), Some(21), Some(i32::MAX)] {
        assert!(openai::validate_completion_logprobs(value).is_err());
        assert!(vllm::validate_completion_logprobs(value).is_err());
    }
    assert_eq!(
        vllm::validate_completion_logprobs(Some(-1)).unwrap_err(),
        ValidationError {
            field: "logprobs",
            message: "must be in [0, 20]"
        },
    );
}

#[test]
fn all_request_traits_preserve_unspecified_values() {
    let chat = CreateChatCompletionRequest::default();
    let completion = CreateCompletionRequest::default();
    let chat_before = chat.clone();
    let completion_before = completion.clone();
    for validate in CHAT_VALIDATORS {
        assert!(validate(&chat).is_ok());
    }
    for validate in COMPLETION_VALIDATORS {
        assert!(validate(&completion).is_ok());
    }
    assert_eq!(chat, chat_before);
    assert_eq!(completion, completion_before);
}

#[test]
fn request_traits_select_engine_overrides_without_running_baseline_first() {
    let mut chat = CreateChatCompletionRequest::default();
    let mut completion = CreateCompletionRequest::default();
    chat.top_p = Some(0.0);
    completion.top_p = Some(0.0);
    assert!(OpenAIValidation::validate(&chat).is_ok());
    assert!(OpenAIValidation::validate(&completion).is_ok());
    for validate in &CHAT_VALIDATORS[1..] {
        assert_eq!(validate(&chat).unwrap_err().field, "top_p");
    }
    for validate in &COMPLETION_VALIDATORS[1..] {
        assert_eq!(validate(&completion).unwrap_err().field, "top_p");
    }
    chat.top_p = Some(0.5);
    completion.top_p = Some(0.5);
    chat.temperature = Some(3.0);
    completion.temperature = Some(3.0);
    assert!(SglangValidation::validate(&chat).is_ok());
    assert!(SglangValidation::validate(&completion).is_ok());
    for validate in &CHAT_VALIDATORS[..2] {
        assert_eq!(validate(&chat).unwrap_err().field, "temperature");
    }
    for validate in &COMPLETION_VALIDATORS[..2] {
        assert_eq!(validate(&completion).unwrap_err().field, "temperature");
    }
    completion.temperature = None;
    completion.logprobs = Some(10);
    assert_eq!(
        OpenAIValidation::validate(&completion).unwrap_err().field,
        "logprobs"
    );
    assert!(VllmValidation::validate(&completion).is_ok());
    completion.logprobs = Some(21);
    assert_eq!(
        VllmValidation::validate(&completion).unwrap_err().field,
        "logprobs"
    );
    assert!(SglangValidation::validate(&completion).is_ok()); // Count validation is uncovered.
}

#[test]
fn all_traits_preserve_first_error_order_and_leave_requests_unchanged() {
    for validate in CHAT_VALIDATORS {
        let mut request = CreateChatCompletionRequest::default();
        request.temperature = Some(-1.0);
        request.top_p = Some(-1.0);
        request.frequency_penalty = Some(3.0);
        request.presence_penalty = Some(3.0);
        let before = request.clone();
        assert_eq!(validate(&request).unwrap_err().field, "temperature");
        assert_eq!(request, before);
        request.temperature = None;
        assert_eq!(validate(&request).unwrap_err().field, "top_p");
        request.top_p = None;
        assert_eq!(validate(&request).unwrap_err().field, "frequency_penalty");
        request.frequency_penalty = None;
        assert_eq!(validate(&request).unwrap_err().field, "presence_penalty");
        request.presence_penalty = None;
        assert!(validate(&request).is_ok());
    }
    for (index, validate) in COMPLETION_VALIDATORS.into_iter().enumerate() {
        let mut request = CreateCompletionRequest::default();
        request.temperature = Some(-1.0);
        request.top_p = Some(-1.0);
        request.frequency_penalty = Some(3.0);
        request.presence_penalty = Some(3.0);
        request.logprobs = Some(21);
        let before = request.clone();
        assert_eq!(validate(&request).unwrap_err().field, "temperature");
        assert_eq!(request, before);
        request.temperature = None;
        assert_eq!(validate(&request).unwrap_err().field, "top_p");
        request.top_p = None;
        assert_eq!(validate(&request).unwrap_err().field, "frequency_penalty");
        request.frequency_penalty = None;
        assert_eq!(validate(&request).unwrap_err().field, "presence_penalty");
        request.presence_penalty = None;
        if index < 2 {
            assert_eq!(validate(&request).unwrap_err().field, "logprobs");
        } else {
            assert!(validate(&request).is_ok());
        }
    }
}

#[test]
fn library_diagnostics_do_not_claim_backend_error_shapes() {
    let error = sglang::validate_temperature(Some(f32::NAN)).unwrap_err();
    assert_eq!(error.field, "temperature");
    assert_eq!(
        error.to_string(),
        "temperature: must be finite and nonnegative"
    );
    assert!(std::error::Error::source(&error).is_none());
    assert_eq!(
        openai::validate_top_p(Some(2.0)).unwrap_err().to_string(),
        "top_p: must be finite and in [0, 1]",
    );
    for validate in [vllm::validate_top_p, sglang::validate_top_p] {
        assert_eq!(
            validate(Some(0.0)).unwrap_err().to_string(),
            "top_p: must be finite and in (0, 1]"
        );
    }
}

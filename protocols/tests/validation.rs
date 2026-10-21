// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_protocols::validation::{ValidationError, dynamo::*};

type FloatValidator = fn(Option<f32>) -> Result<(), ValidationError>;
type ByteValidator = fn(Option<u8>) -> Result<(), ValidationError>;
type TokenValidator = fn(Option<u32>) -> Result<(), ValidationError>;

#[test]
/// Covers inclusive endpoints, their adjacent floats, omission, and non-finite errors.
fn inclusive_float_bounds_and_diagnostics() {
    let rules: [(FloatValidator, &str, f32, f32); 4] = [
        (validate_temperature, "Temperature", 0.0, 2.0),
        (validate_frequency_penalty, "Frequency penalty", -2.0, 2.0),
        (validate_presence_penalty, "Presence penalty", -2.0, 2.0),
        (validate_min_p, "Min_p", 0.0, 1.0),
    ];
    for (validate, label, min, max) in rules {
        for value in [None, Some(min), Some(max), Some(0.5), Some(-0.0)] {
            assert_eq!(validate(value), Ok(()), "{label}: {value:?}");
        }
        for value in [
            min.next_down(),
            max.next_up(),
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NAN,
        ] {
            assert_eq!(
                validate(Some(value)).unwrap_err().to_string(),
                format!("{label} must be between {min} and {max}, got {value}"),
            );
        }
    }
}

#[test]
/// Keeps top_p's exclusive zero bound distinct from other probability fields.
fn top_p_excludes_zero_and_non_finite_values() {
    for value in [None, Some(0.0_f32.next_up()), Some(0.5), Some(1.0)] {
        assert_eq!(validate_top_p(value), Ok(()));
    }
    for value in [
        -0.0,
        0.0,
        0.0_f32.next_down(),
        1.0_f32.next_up(),
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NAN,
    ] {
        assert_eq!(
            validate_top_p(Some(value)).unwrap_err().to_string(),
            format!("Top_p must be between 0 and 1, got {value}"),
        );
    }
}

#[test]
/// Locks in the existing comparison semantics for ordinary floats and typed NaN.
fn repetition_penalty_preserves_typed_nan_acceptance() {
    for value in [
        None,
        Some(0.0_f32.next_up()),
        Some(1.0),
        Some(2.0),
        Some(f32::NAN),
    ] {
        assert_eq!(validate_repetition_penalty(value), Ok(()));
    }
    for value in [
        -0.0,
        0.0,
        0.0_f32.next_down(),
        2.0_f32.next_up(),
        f32::NEG_INFINITY,
        f32::INFINITY,
    ] {
        assert_eq!(
            validate_repetition_penalty(Some(value))
                .unwrap_err()
                .to_string(),
            format!("Repetition penalty must be between 0 and 2, got {value}"),
        );
    }
}

#[test]
/// Checks both accepted sentinel values and the signed integer extremes.
fn top_k_preserves_both_sentinels() {
    for value in [None, Some(-1), Some(0), Some(1), Some(i32::MAX)] {
        assert_eq!(validate_top_k(value), Ok(()));
    }
    for value in [-2, i32::MIN] {
        assert_eq!(
            validate_top_k(Some(value)).unwrap_err().to_string(),
            "Top_k must be null or greater than or equal to -1",
        );
    }
}

#[test]
/// Checks choice and logprob limits with their exact field-specific diagnostics.
fn choice_and_logprob_boundaries_preserve_dynamo_limits() {
    let rules: [(ByteValidator, &str, u8, u8); 3] = [
        (validate_n, "n", 1, 128),
        (validate_top_logprobs, "Top_logprobs", 0, 20),
        (validate_logprobs, "Logprobs", 0, 20),
    ];
    for (validate, label, min, max) in rules {
        for value in [None, Some(min), Some(max), Some(1)] {
            assert_eq!(validate(value), Ok(()));
        }
        for value in [max + 1, u8::MAX] {
            assert_eq!(
                validate(Some(value)).unwrap_err().to_string(),
                format!("{label} must be between {min} and {max}, got {value}"),
            );
        }
    }
    assert_eq!(
        validate_n(Some(0)).unwrap_err().to_string(),
        "n must be between 1 and 128, got 0",
    );
    // Completion logprobs intentionally accepts values above the legacy cap of 5.
    assert_eq!(validate_logprobs(Some(6)), Ok(()));
}

#[test]
/// Distinguishes each token field's zero failure from its upper-limit failure.
fn token_limits_keep_field_specific_zero_and_upper_bound_errors() {
    let rules: [(TokenValidator, &str); 2] = [
        (validate_max_tokens, "Max tokens"),
        (validate_max_completion_tokens, "Max completion tokens"),
    ];
    for (validate, label) in rules {
        for value in [None, Some(1), Some(1_048_576)] {
            assert_eq!(validate(value), Ok(()));
        }
        assert_eq!(
            validate(Some(0)).unwrap_err().to_string(),
            format!("{label} must be greater than 0, got 0"),
        );
        for value in [1_048_577, u32::MAX] {
            assert_eq!(
                validate(Some(value)).unwrap_err().to_string(),
                format!("{label} must not exceed 1048576, got {value}"),
            );
        }
    }
}

#[test]
/// Exercises actual request parsing before opt-in checks, without adding normalization.
fn parsing_and_validation_remain_separate() {
    use dynamo_protocols::types::{CreateChatCompletionRequest, CreateCompletionRequest};
    use serde_json::json;

    for fields in [
        json!({}),
        json!({"temperature": null, "top_p": null, "n": null}),
    ] {
        let mut chat = json!({"model": "test", "messages": []});
        let mut completion = json!({"model": "test", "prompt": "hello"});
        for (key, value) in fields.as_object().unwrap() {
            chat[key] = value.clone();
            completion[key] = value.clone();
        }
        let chat: CreateChatCompletionRequest = serde_json::from_value(chat).unwrap();
        let completion: CreateCompletionRequest = serde_json::from_value(completion).unwrap();
        for (temperature, top_p, n) in [
            (chat.temperature, chat.top_p, chat.n),
            (completion.temperature, completion.top_p, completion.n),
        ] {
            assert_eq!((temperature, top_p, n), (None, None, None));
            validate_temperature(temperature).unwrap();
            validate_top_p(top_p).unwrap();
            validate_n(n).unwrap();
        }
    }

    // Parsing admits representable values without executing the opt-in rules.
    let request: CreateChatCompletionRequest = serde_json::from_value(json!({
        "model": "test", "messages": [], "temperature": 3, "top_p": 0, "n": 0,
    }))
    .unwrap();
    assert!(validate_temperature(request.temperature).is_err());
    assert!(validate_top_p(request.top_p).is_err());
    assert!(validate_n(request.n).is_err());
    assert_eq!(request.temperature, Some(3.0));
    assert_eq!(request.top_p, Some(0.0));
    assert_eq!(request.n, Some(0));

    for n in [json!(-1), json!(256), json!(1.5), json!("1")] {
        assert!(
            serde_json::from_value::<CreateChatCompletionRequest>(json!({
                "model": "test", "messages": [], "n": n,
            }))
            .is_err(),
        );
    }
}

#[test]
/// Checks the standard error contract used by downstream text-only adaptation.
fn errors_have_no_transport_prefix_or_source() {
    let error = validate_temperature(Some(3.0)).unwrap_err();
    let standard_error: &dyn std::error::Error = &error;
    assert_eq!(
        standard_error.to_string(),
        "Temperature must be between 0 and 2, got 3"
    );
    assert!(standard_error.source().is_none());
}

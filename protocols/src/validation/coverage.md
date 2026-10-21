# Coverage and caller contract

These opt-in operations implement a **partial library contract** for the owned
`CreateChatCompletionRequest` and `CreateCompletionRequest` types. They do not
replace a server's full request validator. Parsing and builders do not invoke them.

## Request composition

Each trait checks explicit values in this fixed library order and returns the
first failure:

| Position | OpenAI | vLLM | SGLang |
| --- | --- | --- | --- |
| 1. temperature | finite [0, 2] | reuse OpenAI | finite >= 0 |
| 2. top_p | finite [0, 1] | shared predicate (0, 1] | shared predicate (0, 1] |
| 3. frequency_penalty | finite [-2, 2] | reuse OpenAI | reuse OpenAI |
| 4. presence_penalty | finite [-2, 2] | reuse OpenAI | reuse OpenAI |
| 5. completion logprobs | None or 0..=5 | None or 0..=20 | uncovered |
| 6. n | None or 1..=128 | uncovered | uncovered |
| 7. chat top_logprobs / completion best_of | None or 0..=20 | uncovered | uncovered |

Position 5 applies only to legacy completions. Positions 6 and 7 are OpenAI-only
checks appended after the existing checks. Chat's boolean `logprobs` relationship
and completion best_of relationships to n/streaming are uncovered.
The order is the library's declared composition;
it does not claim to reproduce an engine's complete first-error order.

All float field helpers accept `None` without inserting defaults. For these
existing request fields, omission and explicit null already collapse to `None`.
This boundary cannot reproduce an engine HTTP model that rejects null. Explicit
float checks operate on Rust `f32`, including typed nonfinite inputs, rather than
all Python floats or JSON coercions. Validation does not mutate any request field.

`shared::is_valid_min_p` separately checks finite [0, 1]. The owned request types
do not have `min_p`, `top_k`, or `repetition_penalty` fields; their trait
implementations do not invent them. Engine entry points re-export the shared
`min_p` and exclusive-zero `top_p` predicates. Only these numeric predicates,
not their framework diagnostics, are claimed equivalent. The engine namespaces
also expose `validate_min_p(Option<f32>)` adapters: None passes, finite [0, 1]
passes, and other explicit values return a library error. Their optional input
does not establish matching engine HTTP-null admission.

The library's `ValidationError` identifies the field, its constraint, and the
rejected explicit number in `value: String`. The value uses Rust Display of the
parsed number, including NaN and infinities, rather than the original JSON text.
Formatting and allocation occur only on failure; None does not produce an error.
Its display is `field: message`; it contains no transport status, backend error
metadata, or server-specific numeric formatting. Consumers such as Dynamo can
use the Result helpers and one local adapter for `field` and `value` while
retaining their existing diagnostics and error chains.

## Additional numeric operations and local policy boundaries

The current official references explicitly provide this numeric metadata,
rechecked on 2026-10-21; these are documented bounds, not inferred from field names:

| Operation | Endpoint/reference | Explicit minimum | Explicit maximum |
| --- | --- | --- | --- |
| n | [Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) and [Completions](https://developers.openai.com/api/reference/resources/completions/methods/create) | 1 | 128 |
| best_of count | [Completions](https://developers.openai.com/api/reference/resources/completions/methods/create) | 0 | 20 |
| top_logprobs count | [Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) | 0 | 20 |

The OpenAI count helpers take the existing `Option<u8>` representation and
validate only these bounds. They do not insert defaults or check relationships.
In particular, the OpenAI best_of prose says greater than n; Dynamo's local
greater-than-or-equal relationship is not part of this extracted count operation.
Batch totals and greedy sampling interactions also remain local. vLLM has
configurable n/logprob limits; SGLang equivalence has not been established, so
neither engine re-exports these three complete count validators.

`openai::validate_logit_bias_value(f32)` checks the documented finite [-100,100]
numeric range after the caller's precision conversion. It returns the existing
`ValidationError` with field `logit_bias` and the rejected f32's Rust Display.
It does not validate a map or run from the request traits. The caller must retain
JSON numeric admission, token-ID restrictions, iteration order and per-entry
error precedence. Dynamo's `as_f64() as f32` conversion can round a value just
outside the f64 bounds onto an accepted f32 boundary; extraction must preserve
that order. Original JSON values and map keys are never mutated by this helper.

Whole-map extraction was rejected: OpenAI's tokenizer-ID documentation does not
establish Dynamo's u32 lexical/range restriction. The pinned
[vLLM conversion](https://github.com/vllm-project/vllm/blob/6d1daa9fe53f39a7b494cf6c8e433a73eec78a76/vllm/sampling_params.py#L488)
parses Python integers and clamps bias values instead of rejecting their range.
The pinned [SGLang check](https://github.com/sgl-project/sglang/blob/5569b14d26ffd94a8326889eca15a153e8f1b68b/python/sglang/srt/sampling/sampling_params.py#L272)
uses vocabulary size for token IDs and does not establish the same value bounds.
Neither engine reuses this rejecting value operation in the current slice.

Stop validation also remains outside this extraction. OpenAI documents at most
four text sequences. Dynamo's 32-item string/token-ID limit, empty-array rejection,
and typed count errors differ. Pinned vLLM rejects empty strings but permits an
empty stop list and has a configured count limit; SGLang's normalization does not
establish the same empty-string rejection. A common full operation is not proven.

## Dedicated-server usage

Call only the selected contract. Do not run the baseline and an override in
sequence: their accepted sets can differ.

```rust
use dynamo_protocols::{
    normalization,
    types::CreateCompletionRequest,
    validation::{openai::OpenAIValidation, vllm::VllmValidation},
};

let mut request = CreateCompletionRequest::default();
request.logprobs = Some(10);
normalization::vllm::normalize_completion_logprobs(&mut request.logprobs);
assert!(VllmValidation::validate(&request).is_ok());
assert!(OpenAIValidation::validate(&request).is_err());
assert_eq!(request.logprobs, Some(10));
```

This example exercises only the covered operations; the default request still
needs a server's prompt, model, and other checks before dispatch.

## Deliberate completion-logprobs policy

The vLLM helper implements the fixed library policy from
[DEP #16992](https://github.com/ai-dynamo/dynamo/issues/16992):
`None` or 0..=20, rejecting -1, with no model context. It replaces the OpenAI
0..=5 operation in `VllmValidation`; it does not run after that baseline check.

This intentionally differs from actual vLLM acceptance. vLLM recognizes -1
for full vocabulary and uses configured `max_logprobs`, whose default is 20.
The fixed policy can reject backend-supported values or accept values rejected
by a stricter configured cap. The field helper
takes `Option<i32>` so direct callers can check negative values; the owned
request remains `Option<u8>`, so parsing rejects negatives before its trait runs.
No public request representation or schema changes.

## Normalization coverage

`normalization::openai::normalize_completion_logprobs` preserves the existing
`Option<u8>` field. Both engine modules explicitly reuse this operation at the
same typed request boundary, before their later conversion. It is not a shared
non-OpenAI extension, so no `normalization::shared` operation is introduced.

| Input representation | Parsed value | Normalizer output | Current serialized representation |
| --- | --- | --- | --- |
| Missing key | None | None | key omitted |
| Explicit null | None | None | key omitted |
| Explicit zero | Some(0) | Some(0) | numeric zero |
| Other representable count, valid or invalid | Some(value) | Some(value) | same number |

The preservation rule is deliberate: validation supplies no effective default,
and normalization must not clamp a count or erase an invalid explicit value.
Call it after parsing and before the chosen validation operation. Repeated calls
leave the same state; it requires no context and cannot fail or partially mutate.
The later engine request-conversion layer owns enablement and internal count
selection. This no-op does not assert that those later representations match.

No other normalizers are covered. In particular, temperature/penalty defaults,
model generation configuration, token-limit resolution, and greedy/sentinel
rewrites remain with the caller that has the necessary context and ordering.
No universal normalize-then-validate lifecycle is imposed on other fields.

## Uncovered checks

All unlisted fields and cross-field rules remain uncovered, including engine
n/best_of/top_logprobs counts, chat logprob relationships, token limits, penalties
absent from these owned types, required prompt/messages/model content, tools, stream options,
guided decoding, model capabilities, and backend dispatch checks. SGLang count
rules remain unclassified. Trait success does not validate these concerns.
Full logit-bias map admission and stop validation remain uncovered.
Dynamo-specific constants, validators, normalization, and request composition
remain in Dynamo; this crate has no Dynamo policy namespace.

## Sources and comparison layer

OpenAI numeric bounds are documented in the
[Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
and the [legacy Completions reference](https://developers.openai.com/api/reference/resources/completions/methods/create),
read on 2026-10-21. These establish documented bounds, not hosted-server probes
or exact HTTP diagnostics.

Engine predicate assessment uses ordinary sampling for chat/completion routes,
with these historical source pins, not a claim about later engine revisions:

- vLLM `6d1daa9fe53f39a7b494cf6c8e433a73eec78a76`:
  [numeric checks and greedy order](https://github.com/vllm-project/vllm/blob/6d1daa9fe53f39a7b494cf6c8e433a73eec78a76/vllm/sampling_params.py#L556),
  [chat conversion](https://github.com/vllm-project/vllm/blob/6d1daa9fe53f39a7b494cf6c8e433a73eec78a76/vllm/entrypoints/openai/chat_completion/protocol.py#L698),
  [completion conversion](https://github.com/vllm-project/vllm/blob/6d1daa9fe53f39a7b494cf6c8e433a73eec78a76/vllm/entrypoints/openai/completion/protocol.py#L331).
- SGLang `5569b14d26ffd94a8326889eca15a153e8f1b68b`:
  [numeric checks and initialization](https://github.com/sgl-project/sglang/blob/5569b14d26ffd94a8326889eca15a153e8f1b68b/python/sglang/srt/sampling/sampling_params.py#L168),
  [request declarations and conversion](https://github.com/sgl-project/sglang/blob/5569b14d26ffd94a8326889eca15a153e8f1b68b/python/sglang/srt/entrypoints/openai/protocol.py#L351),
  [completion conversion](https://github.com/sgl-project/sglang/blob/5569b14d26ffd94a8326889eca15a153e8f1b68b/python/sglang/srt/entrypoints/openai/serving_completions.py#L116),
  [construction and verification](https://github.com/sgl-project/sglang/blob/5569b14d26ffd94a8326889eca15a153e8f1b68b/python/sglang/srt/managers/tokenizer_manager.py#L1535).

The fixed vLLM logprobs policy is separately selected by the DEP. Its documented
departure refers to [vLLM v0.28.0 count checks](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/sampling_params.py)
and [configured default](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/config/model.py).
It is not inferred from the pinned predicate comparison.

Raw engine checks, HTTP admission, preprocessing, and library representation
are different layers. In particular, vLLM floors tiny positive temperatures and
later rewrites greedy sampling parameters; SGLang performs a different greedy
rewrite before its checks. Those transformations are not implemented here.

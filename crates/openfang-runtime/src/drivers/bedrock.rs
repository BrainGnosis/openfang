//! AWS Bedrock Converse API driver.
//!
//! Native implementation of the Bedrock Converse API using the official
//! `aws-sdk-bedrockruntime` crate. Handles SigV4 authentication, credential
//! chain resolution, streaming via binary event-stream, and type conversion
//! between Bedrock SDK types and OpenFang's `LlmDriver` trait.
//!
//! Uses `tokio::sync::OnceCell` for lazy client initialization because
//! `create_driver()` in mod.rs is synchronous but AWS config loading is async.

use crate::llm_driver::{CompletionRequest, CompletionResponse, LlmDriver, LlmError, StreamEvent};
use async_trait::async_trait;
use aws_sdk_bedrockruntime::types::{
    ContentBlock as BedrockContentBlock, ConversationRole, InferenceConfiguration, Message as BedrockMessage,
    SystemContentBlock, Tool, ToolConfiguration, ToolInputSchema, ToolResultBlock,
    ToolResultContentBlock, ToolSpecification, ToolUseBlock,
};
use aws_sdk_bedrockruntime::Client;
use openfang_types::message::{
    ContentBlock, Message, MessageContent, Role, StopReason, TokenUsage,
};
use openfang_types::tool::{ToolCall, ToolDefinition};
use tokio::sync::OnceCell;
use tracing::{debug, warn};

/// AWS Bedrock Converse API driver.
///
/// Uses the official AWS SDK for Rust which handles SigV4 signing, credential
/// chain (env vars, profiles, IMDS), and streaming deserialization automatically.
pub struct BedrockDriver {
    /// Lazily initialized Bedrock client. Uses OnceCell because the constructor
    /// must be synchronous (create_driver() is not async) but aws_config loading
    /// requires async.
    client: OnceCell<Client>,
    /// AWS region for the Bedrock endpoint.
    region: String,
}

impl BedrockDriver {
    /// Create a new Bedrock driver. Constructor is intentionally synchronous.
    /// The AWS SDK client is lazily initialized on first use.
    pub fn new(region: String) -> Self {
        Self {
            client: OnceCell::new(),
            region,
        }
    }

    /// Get or initialize the Bedrock client.
    async fn client(&self) -> &Client {
        self.client
            .get_or_init(|| async {
                let config = aws_config::defaults(aws_config::BehaviorVersion::latest())
                    .region(aws_config::Region::new(self.region.clone()))
                    .load()
                    .await;
                debug!(region = %self.region, "Initialized Bedrock client");
                Client::new(&config)
            })
            .await
    }
}

/// Extract the Bedrock model ID from the OpenFang model string.
///
/// Handles:
/// - "bedrock/anthropic.claude-opus-4-6" → "anthropic.claude-opus-4-6"
/// - "anthropic.claude-opus-4-6" → "anthropic.claude-opus-4-6" (no prefix)
/// - "arn:aws:bedrock:..." → passed through as-is (cross-region inference profiles)
fn extract_model_id(model: &str) -> &str {
    model.strip_prefix("bedrock/").unwrap_or(model)
}

/// Convert a `serde_json::Value` to an `aws_smithy_types::Document`.
///
/// The AWS SDK `Document` type does not implement `From<serde_json::Value>` (serde
/// support is behind the unstable `aws_sdk_unstable` cfg flag), so we do the
/// recursive conversion by hand.
fn json_to_document(v: &serde_json::Value) -> aws_smithy_types::Document {
    use aws_smithy_types::Document;
    match v {
        serde_json::Value::Null => Document::Null,
        serde_json::Value::Bool(b) => Document::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_u64() {
                Document::Number(aws_smithy_types::Number::PosInt(i))
            } else if let Some(i) = n.as_i64() {
                Document::Number(aws_smithy_types::Number::NegInt(i))
            } else if let Some(f) = n.as_f64() {
                Document::Number(aws_smithy_types::Number::Float(f))
            } else {
                Document::Null
            }
        }
        serde_json::Value::String(s) => Document::String(s.clone()),
        serde_json::Value::Array(arr) => {
            Document::Array(arr.iter().map(json_to_document).collect())
        }
        serde_json::Value::Object(map) => {
            Document::Object(map.iter().map(|(k, v)| (k.clone(), json_to_document(v))).collect())
        }
    }
}

/// Convert an `aws_smithy_types::Document` to a `serde_json::Value`.
#[allow(unreachable_patterns)] // Document and Number are #[non_exhaustive] — wildcards needed for forward compat
fn document_to_json(doc: &aws_smithy_types::Document) -> serde_json::Value {
    use aws_smithy_types::Document;
    match doc {
        Document::Null => serde_json::Value::Null,
        Document::Bool(b) => serde_json::Value::Bool(*b),
        Document::Number(n) => match n {
            aws_smithy_types::Number::PosInt(i) => serde_json::json!(*i),
            aws_smithy_types::Number::NegInt(i) => serde_json::json!(*i),
            aws_smithy_types::Number::Float(f) => {
                serde_json::Value::Number(serde_json::Number::from_f64(*f).unwrap_or_else(|| serde_json::Number::from(0)))
            }
            _ => serde_json::Value::Null,
        },
        Document::String(s) => serde_json::Value::String(s.clone()),
        Document::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(document_to_json).collect())
        }
        Document::Object(map) => {
            serde_json::Value::Object(map.iter().map(|(k, v)| (k.clone(), document_to_json(v))).collect())
        }
        _ => serde_json::Value::Null,
    }
}

/// Ensure a `serde_json::Value` is a JSON object.
///
/// The Bedrock API requires tool inputs to be JSON objects. This normalizes
/// stringified or null values to proper objects.
fn ensure_object(v: &serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::Object(_) => v.clone(),
        serde_json::Value::String(s) => match serde_json::from_str::<serde_json::Value>(s) {
            Ok(parsed) if parsed.is_object() => parsed,
            _ => serde_json::json!({}),
        },
        _ => serde_json::json!({}),
    }
}

/// Map a Bedrock stop reason to OpenFang's StopReason.
fn map_stop_reason(reason: &aws_sdk_bedrockruntime::types::StopReason) -> StopReason {
    use aws_sdk_bedrockruntime::types::StopReason as BR;
    match reason {
        BR::EndTurn => StopReason::EndTurn,
        BR::ToolUse => StopReason::ToolUse,
        BR::MaxTokens => StopReason::MaxTokens,
        BR::StopSequence => StopReason::StopSequence,
        BR::ContentFiltered => {
            warn!("Bedrock: content was filtered by content policy");
            StopReason::EndTurn
        }
        BR::GuardrailIntervened => {
            warn!("Bedrock: guardrail intervened");
            StopReason::EndTurn
        }
        _ => {
            // #[non_exhaustive] — handle unknown future variants
            warn!(reason = ?reason, "Bedrock: unknown stop reason, treating as EndTurn");
            StopReason::EndTurn
        }
    }
}

/// Map an AWS SDK error to an OpenFang LlmError.
fn map_sdk_error<E: std::fmt::Display + std::fmt::Debug>(
    err: aws_sdk_bedrockruntime::error::SdkError<E>,
) -> LlmError {
    use aws_sdk_bedrockruntime::error::SdkError;
    match &err {
        SdkError::ServiceError(service_err) => {
            let msg = format!("{}", service_err.err());
            if msg.contains("ThrottlingException") || msg.contains("Too many requests") {
                LlmError::RateLimited {
                    retry_after_ms: 5000,
                }
            } else if msg.contains("AccessDeniedException") {
                LlmError::AuthenticationFailed(msg)
            } else if msg.contains("ModelNotReadyException")
                || msg.contains("ServiceUnavailableException")
            {
                LlmError::Overloaded {
                    retry_after_ms: 5000,
                }
            } else if msg.contains("ResourceNotFoundException")
                || msg.contains("ModelNotFoundException")
            {
                LlmError::ModelNotFound(msg)
            } else if msg.contains("ValidationException") {
                // ValidationException = malformed request (e.g., unsupported feature for model)
                // Map to Api, NOT Parse — it's a request error, not a response parse error
                LlmError::Api { status: 400, message: msg }
            } else {
                LlmError::Api { status: 0, message: msg }
            }
        }
        SdkError::TimeoutError(_) => LlmError::Http("AWS SDK request timeout".to_string()),
        SdkError::DispatchFailure(df) => {
            LlmError::Http(format!("AWS SDK connection failure: {:?}", df))
        }
        _ => {
            // #[non_exhaustive]
            LlmError::Http(format!("AWS SDK error: {}", err))
        }
    }
}

/// Convert OpenFang messages to Bedrock messages.
///
/// Filters out System role messages (system prompt is handled separately).
/// Converts content blocks between the two type systems.
fn convert_messages(messages: &[Message]) -> Vec<BedrockMessage> {
    messages
        .iter()
        .filter(|m| m.role != Role::System)
        .filter_map(|msg| {
            let role = match msg.role {
                Role::User => ConversationRole::User,
                Role::Assistant => ConversationRole::Assistant,
                Role::System => return None, // filtered above, but be safe
            };

            let content_blocks = match &msg.content {
                MessageContent::Text(text) => {
                    vec![BedrockContentBlock::Text(text.clone())]
                }
                MessageContent::Blocks(blocks) => blocks
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text { text, .. } => {
                            Some(BedrockContentBlock::Text(text.clone()))
                        }
                        ContentBlock::Image { media_type, data } => {
                            // Bedrock expects raw bytes, not base64 string
                            match base64::Engine::decode(
                                &base64::engine::general_purpose::STANDARD,
                                data,
                            ) {
                                Ok(bytes) => {
                                    let format = match media_type.as_str() {
                                        "image/png" => {
                                            aws_sdk_bedrockruntime::types::ImageFormat::Png
                                        }
                                        "image/gif" => {
                                            aws_sdk_bedrockruntime::types::ImageFormat::Gif
                                        }
                                        "image/webp" => {
                                            aws_sdk_bedrockruntime::types::ImageFormat::Webp
                                        }
                                        _ => aws_sdk_bedrockruntime::types::ImageFormat::Jpeg,
                                    };
                                    Some(BedrockContentBlock::Image(
                                        aws_sdk_bedrockruntime::types::ImageBlock::builder()
                                            .format(format)
                                            .source(
                                                aws_sdk_bedrockruntime::types::ImageSource::Bytes(
                                                    aws_smithy_types::Blob::new(bytes),
                                                ),
                                            )
                                            .build()
                                            .ok()?,
                                    ))
                                }
                                Err(e) => {
                                    warn!(error = %e, "Failed to decode base64 image data");
                                    None
                                }
                            }
                        }
                        ContentBlock::ToolUse {
                            id, name, input, ..
                        } => Some(BedrockContentBlock::ToolUse(
                            ToolUseBlock::builder()
                                .tool_use_id(id.clone())
                                .name(name.clone())
                                .input(json_to_document(&ensure_object(input)))
                                .build()
                                .ok()?,
                        )),
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                            ..
                        } => {
                            let mut builder = ToolResultBlock::builder()
                                .tool_use_id(tool_use_id.clone())
                                .content(ToolResultContentBlock::Text(content.clone()));
                            if *is_error {
                                builder = builder
                                    .status(aws_sdk_bedrockruntime::types::ToolResultStatus::Error);
                            }
                            Some(BedrockContentBlock::ToolResult(builder.build().ok()?))
                        }
                        ContentBlock::Thinking { .. } | ContentBlock::Unknown => None,
                    })
                    .collect(),
            };

            if content_blocks.is_empty() {
                return None;
            }

            BedrockMessage::builder()
                .role(role)
                .set_content(Some(content_blocks))
                .build()
                .ok()
        })
        .collect()
}

/// Convert OpenFang tool definitions to Bedrock tool configuration.
fn convert_tool_config(tools: &[ToolDefinition]) -> Option<ToolConfiguration> {
    if tools.is_empty() {
        return None;
    }

    let bedrock_tools: Vec<Tool> = tools
        .iter()
        .filter_map(|t| {
            let input_schema = ToolInputSchema::Json(
                json_to_document(&t.input_schema),
            );
            let spec = ToolSpecification::builder()
                .name(&t.name)
                .description(&t.description)
                .input_schema(input_schema)
                .build()
                .ok()?;
            Some(Tool::ToolSpec(spec))
        })
        .collect();

    if bedrock_tools.is_empty() {
        return None;
    }

    ToolConfiguration::builder()
        .set_tools(Some(bedrock_tools))
        .build()
        .ok()
}

/// Convert a Bedrock Converse response to OpenFang's CompletionResponse.
fn convert_response(
    output: &aws_sdk_bedrockruntime::operation::converse::ConverseOutput,
) -> CompletionResponse {
    let mut content = Vec::new();
    let mut tool_calls = Vec::new();

    if let Some(msg) = output.output().and_then(|o| {
        if let aws_sdk_bedrockruntime::types::ConverseOutput::Message(m) = o {
            Some(m)
        } else {
            None
        }
    }) {
        for block in msg.content() {
            match block {
                BedrockContentBlock::Text(text) => {
                    content.push(ContentBlock::Text {
                        text: text.clone(),
                        provider_metadata: None,
                    });
                }
                BedrockContentBlock::ToolUse(tu) => {
                    let input = document_to_json(tu.input());
                    let id = tu.tool_use_id().to_string();
                    let name = tu.name().to_string();
                    content.push(ContentBlock::ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        input: input.clone(),
                        provider_metadata: None,
                    });
                    tool_calls.push(ToolCall {
                        id,
                        name,
                        input,
                    });
                }
                _ => {
                    // #[non_exhaustive] — skip unknown block types
                }
            }
        }
    }

    let stop_reason = map_stop_reason(output.stop_reason());

    let usage = output
        .usage()
        .map(|u| TokenUsage {
            input_tokens: u.input_tokens() as u64,
            output_tokens: u.output_tokens() as u64,
        })
        .unwrap_or_default();

    CompletionResponse {
        content,
        stop_reason,
        tool_calls,
        usage,
    }
}

#[async_trait]
impl LlmDriver for BedrockDriver {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let model_id = extract_model_id(&request.model);
        debug!(model_id, "Bedrock converse request");

        let bedrock_messages = convert_messages(&request.messages);
        let tool_config = convert_tool_config(&request.tools);

        let inference_config = InferenceConfiguration::builder()
            .max_tokens(request.max_tokens as i32)
            .temperature(request.temperature)
            .build();

        // Retry loop for throttling and service errors
        let max_retries = 3;
        for attempt in 0..=max_retries {
            let mut req = self
                .client()
                .await
                .converse()
                .model_id(model_id)
                .set_messages(Some(bedrock_messages.clone()))
                .inference_config(inference_config.clone());

            // System prompt — Bedrock takes it as a separate parameter, not a message
            if let Some(ref system_text) = request.system {
                req = req.system(SystemContentBlock::Text(system_text.clone()));
            }

            // Tool configuration
            if let Some(ref tc) = tool_config {
                req = req.tool_config(tc.clone());
            }

            match req.send().await {
                Ok(output) => {
                    return Ok(convert_response(&output));
                }
                Err(err) => {
                    let llm_err = map_sdk_error(err);
                    match &llm_err {
                        LlmError::RateLimited { .. } | LlmError::Overloaded { .. }
                            if attempt < max_retries =>
                        {
                            let retry_ms = (attempt + 1) as u64 * 2000;
                            warn!(attempt, retry_ms, "Bedrock rate limited, retrying");
                            tokio::time::sleep(std::time::Duration::from_millis(retry_ms)).await;
                            continue;
                        }
                        _ => return Err(llm_err),
                    }
                }
            }
        }

        Err(LlmError::Api {
            status: 0,
            message: "Bedrock: max retries exceeded".to_string(),
        })
    }

    async fn stream(
        &self,
        request: CompletionRequest,
        tx: tokio::sync::mpsc::Sender<StreamEvent>,
    ) -> Result<CompletionResponse, LlmError> {
        let model_id = extract_model_id(&request.model);
        debug!(model_id, "Bedrock converse_stream request");

        let bedrock_messages = convert_messages(&request.messages);
        let tool_config = convert_tool_config(&request.tools);

        let inference_config = InferenceConfiguration::builder()
            .max_tokens(request.max_tokens as i32)
            .temperature(request.temperature)
            .build();

        // Retry loop for the initial request (not mid-stream failures)
        let max_retries = 3;
        for attempt in 0..=max_retries {
            let mut req = self
                .client()
                .await
                .converse_stream()
                .model_id(model_id)
                .set_messages(Some(bedrock_messages.clone()))
                .inference_config(inference_config.clone());

            if let Some(ref system_text) = request.system {
                req = req.system(SystemContentBlock::Text(system_text.clone()));
            }

            if let Some(ref tc) = tool_config {
                req = req.tool_config(tc.clone());
            }

            let output = match req.send().await {
                Ok(output) => output,
                Err(err) => {
                    let llm_err = map_sdk_error(err);
                    match &llm_err {
                        LlmError::RateLimited { .. } | LlmError::Overloaded { .. }
                            if attempt < max_retries =>
                        {
                            let retry_ms = (attempt + 1) as u64 * 2000;
                            warn!(attempt, retry_ms, "Bedrock stream rate limited, retrying");
                            tokio::time::sleep(std::time::Duration::from_millis(retry_ms)).await;
                            continue;
                        }
                        _ => return Err(llm_err),
                    }
                }
            };

            // Process the event stream
            let mut event_stream = output.stream;

            // Accumulators for building the final response
            let mut blocks: Vec<ContentBlockAccum> = Vec::new();
            let mut stop_reason = StopReason::EndTurn;
            let mut usage = TokenUsage::default();

            loop {
                match event_stream.recv().await {
                    Ok(Some(event)) => {
                        use aws_sdk_bedrockruntime::types::ConverseStreamOutput as CSO;
                        match event {
                            CSO::MessageStart(_) => {
                                // Nothing to do — role is always assistant
                            }
                            CSO::ContentBlockStart(start) => {
                                if let Some(inner) = start.start() {
                                    use aws_sdk_bedrockruntime::types::ContentBlockStart as CBS;
                                    match inner {
                                        CBS::ToolUse(tu) => {
                                            let id =
                                                tu.tool_use_id().to_string();
                                            let name = tu.name().to_string();
                                            if tx
                                                .send(StreamEvent::ToolUseStart {
                                                    id: id.clone(),
                                                    name: name.clone(),
                                                })
                                                .await
                                                .is_err()
                                            {
                                                // Receiver dropped — stop to save tokens
                                                debug!("Stream receiver dropped at ToolUseStart");
                                                drop(event_stream);
                                                return Err(LlmError::Http(
                                                    "Stream cancelled by caller".to_string(),
                                                ));
                                            }
                                            blocks.push(ContentBlockAccum::ToolUse {
                                                id,
                                                name,
                                                input_json: String::new(),
                                            });
                                        }
                                        _ => {
                                            // #[non_exhaustive] — text blocks start with no metadata
                                            blocks.push(ContentBlockAccum::Text(String::new()));
                                        }
                                    }
                                }
                            }
                            CSO::ContentBlockDelta(delta) => {
                                if let Some(inner) = delta.delta() {
                                    use aws_sdk_bedrockruntime::types::ContentBlockDelta as CBD;
                                    match inner {
                                        CBD::Text(ref text_delta) => {
                                            let text = text_delta.clone();
                                            if let Some(ContentBlockAccum::Text(ref mut t)) =
                                                blocks.last_mut()
                                            {
                                                t.push_str(&text);
                                            }
                                            if !text.is_empty()
                                                && tx
                                                    .send(StreamEvent::TextDelta {
                                                        text: text.clone(),
                                                    })
                                                    .await
                                                    .is_err()
                                            {
                                                debug!("Stream receiver dropped at TextDelta");
                                                drop(event_stream);
                                                return Err(LlmError::Http(
                                                    "Stream cancelled by caller".to_string(),
                                                ));
                                            }
                                        }
                                        CBD::ToolUse(tu_delta) => {
                                            let partial =
                                                tu_delta.input().to_string();
                                            if let Some(ContentBlockAccum::ToolUse {
                                                ref mut input_json,
                                                ..
                                            }) = blocks.last_mut()
                                            {
                                                input_json.push_str(&partial);
                                            }
                                            if !partial.is_empty() {
                                                let _ = tx
                                                    .send(StreamEvent::ToolInputDelta {
                                                        text: partial,
                                                    })
                                                    .await;
                                            }
                                        }
                                        _ => {
                                            // #[non_exhaustive] — handle reasoning deltas etc.
                                        }
                                    }
                                }
                            }
                            CSO::ContentBlockStop(stop) => {
                                let block_idx =
                                    stop.content_block_index() as usize;
                                if let Some(ContentBlockAccum::ToolUse {
                                    id,
                                    name,
                                    input_json,
                                }) = blocks.get(block_idx)
                                {
                                    let input: serde_json::Value =
                                        serde_json::from_str(input_json)
                                            .unwrap_or_else(|_| serde_json::json!({}));
                                    let _ = tx
                                        .send(StreamEvent::ToolUseEnd {
                                            id: id.clone(),
                                            name: name.clone(),
                                            input,
                                        })
                                        .await;
                                }
                            }
                            CSO::MessageStop(stop) => {
                                stop_reason = map_stop_reason(stop.stop_reason());
                            }
                            CSO::Metadata(meta) => {
                                if let Some(u) = meta.usage() {
                                    usage = TokenUsage {
                                        input_tokens: u.input_tokens() as u64,
                                        output_tokens: u.output_tokens() as u64,
                                    };
                                }
                            }
                            _ => {
                                // #[non_exhaustive]
                            }
                        }
                    }
                    Ok(None) => {
                        // Stream finished
                        break;
                    }
                    Err(e) => {
                        warn!(error = %e, "Bedrock stream error (mid-stream)");
                        // Mid-stream error — we may have partial data.
                        // Return what we have if any content was accumulated.
                        if !blocks.is_empty() {
                            break;
                        }
                        return Err(LlmError::Http(format!(
                            "Bedrock stream error: {}",
                            e
                        )));
                    }
                }
            }

            // Build final CompletionResponse from accumulated blocks
            let mut content = Vec::new();
            let mut tool_calls = Vec::new();
            for block in blocks {
                match block {
                    ContentBlockAccum::Text(text) => {
                        if !text.is_empty() {
                            content.push(ContentBlock::Text {
                                text,
                                provider_metadata: None,
                            });
                        }
                    }
                    ContentBlockAccum::Thinking(thinking) => {
                        content.push(ContentBlock::Thinking { thinking });
                    }
                    ContentBlockAccum::ToolUse {
                        id,
                        name,
                        input_json,
                    } => {
                        let input: serde_json::Value = serde_json::from_str(&input_json)
                            .unwrap_or_else(|_| serde_json::json!({}));
                        content.push(ContentBlock::ToolUse {
                            id: id.clone(),
                            name: name.clone(),
                            input: input.clone(),
                            provider_metadata: None,
                        });
                        tool_calls.push(ToolCall { id, name, input });
                    }
                }
            }

            let _ = tx
                .send(StreamEvent::ContentComplete { stop_reason, usage })
                .await;

            return Ok(CompletionResponse {
                content,
                stop_reason,
                tool_calls,
                usage,
            });
        }

        Err(LlmError::Api {
            status: 0,
            message: "Bedrock stream: max retries exceeded".to_string(),
        })
    }
}

/// Accumulator for content blocks during streaming.
#[allow(dead_code)] // Thinking variant reserved for extended thinking support
enum ContentBlockAccum {
    Text(String),
    Thinking(String),
    ToolUse {
        id: String,
        name: String,
        input_json: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_model_id_with_prefix() {
        assert_eq!(
            extract_model_id("bedrock/anthropic.claude-opus-4-6"),
            "anthropic.claude-opus-4-6"
        );
    }

    #[test]
    fn test_extract_model_id_without_prefix() {
        assert_eq!(
            extract_model_id("anthropic.claude-opus-4-6"),
            "anthropic.claude-opus-4-6"
        );
    }

    #[test]
    fn test_extract_model_id_arn() {
        let arn = "arn:aws:bedrock:us-east-1:123456:inference-profile/test";
        assert_eq!(extract_model_id(arn), arn);
    }

    #[test]
    fn test_ensure_object_from_object() {
        let obj = serde_json::json!({"key": "value"});
        assert_eq!(ensure_object(&obj), obj);
    }

    #[test]
    fn test_ensure_object_from_string() {
        let stringified = serde_json::Value::String(r#"{"query": "rust"}"#.to_string());
        let result = ensure_object(&stringified);
        assert_eq!(result, serde_json::json!({"query": "rust"}));
    }

    #[test]
    fn test_ensure_object_from_null() {
        assert_eq!(ensure_object(&serde_json::Value::Null), serde_json::json!({}));
    }

    #[test]
    fn test_convert_messages_filters_system() {
        let messages = vec![
            Message {
                role: Role::System,
                content: MessageContent::Text("You are helpful".to_string()),
            },
            Message::user("Hello"),
        ];
        let bedrock_msgs = convert_messages(&messages);
        assert_eq!(bedrock_msgs.len(), 1);
    }

    #[test]
    fn test_convert_messages_text() {
        let messages = vec![
            Message::user("Hello"),
            Message {
                role: Role::Assistant,
                content: MessageContent::Text("Hi there!".to_string()),
            },
        ];
        let bedrock_msgs = convert_messages(&messages);
        assert_eq!(bedrock_msgs.len(), 2);
    }

    #[test]
    fn test_convert_tool_config_empty() {
        assert!(convert_tool_config(&[]).is_none());
    }

    #[test]
    fn test_convert_tool_config_with_tools() {
        let tools = vec![ToolDefinition {
            name: "web_search".to_string(),
            description: "Search the web".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }),
        }];
        let config = convert_tool_config(&tools);
        assert!(config.is_some());
    }

    #[test]
    fn test_map_stop_reason_variants() {
        use aws_sdk_bedrockruntime::types::StopReason as BR;
        assert_eq!(map_stop_reason(&BR::EndTurn), StopReason::EndTurn);
        assert_eq!(map_stop_reason(&BR::ToolUse), StopReason::ToolUse);
        assert_eq!(map_stop_reason(&BR::MaxTokens), StopReason::MaxTokens);
        assert_eq!(
            map_stop_reason(&BR::StopSequence),
            StopReason::StopSequence
        );
        // Bedrock-specific variants map to EndTurn
        assert_eq!(
            map_stop_reason(&BR::ContentFiltered),
            StopReason::EndTurn
        );
        assert_eq!(
            map_stop_reason(&BR::GuardrailIntervened),
            StopReason::EndTurn
        );
    }
}

//! Slack Socket Mode adapter for the OpenFang channel bridge.
//!
//! Uses Slack Socket Mode WebSocket (app token) for receiving events and the
//! Web API (bot token) for sending responses. No external Slack crate.

use crate::types::{
    split_message, ChannelAdapter, ChannelContent, ChannelMessage, ChannelType, ChannelUser,
    LifecycleReaction,
};
use async_trait::async_trait;
use dashmap::DashMap;
use futures::{SinkExt, Stream, StreamExt};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, watch, RwLock};
use tracing::{debug, error, info, warn};
use zeroize::Zeroizing;

const SLACK_API_BASE: &str = "https://slack.com/api";
const MAX_BACKOFF: Duration = Duration::from_secs(60);
const INITIAL_BACKOFF: Duration = Duration::from_secs(1);
const SLACK_MSG_LIMIT: usize = 3000;

/// Slack Socket Mode adapter.
pub struct SlackAdapter {
    /// SECURITY: Tokens are zeroized on drop to prevent memory disclosure.
    app_token: Zeroizing<String>,
    bot_token: Zeroizing<String>,
    client: reqwest::Client,
    allowed_channels: Vec<String>,
    shutdown_tx: Arc<watch::Sender<bool>>,
    shutdown_rx: watch::Receiver<bool>,
    /// Bot's own user ID (populated after auth.test).
    bot_user_id: Arc<RwLock<Option<String>>>,
    /// Threads where the bot was @-mentioned. Maps thread_ts -> last interaction time.
    active_threads: Arc<DashMap<String, Instant>>,
    /// How long to track a thread after last interaction.
    thread_ttl: Duration,
    /// Whether auto-thread-reply is enabled.
    auto_thread_reply: bool,
    /// Whether to unfurl (expand previews for) links in posted messages.
    unfurl_links: bool,
    /// [braingnosis] Cache: user_id → DM channel_id. Used because platform_id is now
    /// user_id (for RBAC) but Slack API calls need channel_id.
    user_dm_channels: Arc<DashMap<String, String>>,
    /// [braingnosis] Cache: user_id → channel_id from last incoming message.
    /// Populated by parse_slack_event, consumed by send/send_in_thread.
    last_channel_for_user: Arc<DashMap<String, String>>,
    /// [braingnosis] Track last reaction shortcode per message_ts, so we can remove it
    /// when `remove_previous` is set. Key: "channel:ts", Value: shortcode.
    last_reaction: Arc<DashMap<String, String>>,
    /// [braingnosis] Cache: user_id → display_name. Resolved via users.info API.
    user_names: Arc<DashMap<String, String>>,
    /// [braingnosis] Rate limit tracker: API method → last call timestamp.
    /// Enforces minimum interval between calls to avoid Slack rate limiting.
    rate_limits: Arc<DashMap<String, Instant>>,
}

impl SlackAdapter {
    pub fn new(
        app_token: String,
        bot_token: String,
        allowed_channels: Vec<String>,
        auto_thread_reply: bool,
        thread_ttl_hours: u64,
        unfurl_links: bool,
    ) -> Self {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        Self {
            app_token: Zeroizing::new(app_token),
            bot_token: Zeroizing::new(bot_token),
            client: reqwest::Client::new(),
            allowed_channels,
            shutdown_tx: Arc::new(shutdown_tx),
            shutdown_rx,
            bot_user_id: Arc::new(RwLock::new(None)),
            active_threads: Arc::new(DashMap::new()),
            thread_ttl: Duration::from_secs(thread_ttl_hours * 3600),
            auto_thread_reply,
            unfurl_links,
            user_dm_channels: Arc::new(DashMap::new()),
            last_channel_for_user: Arc::new(DashMap::new()),
            last_reaction: Arc::new(DashMap::new()),
            user_names: Arc::new(DashMap::new()),
            rate_limits: Arc::new(DashMap::new()),
        }
    }

    /// [braingnosis] Enforce rate limiting for a Slack API method.
    ///
    /// Slack Tier 2 endpoints allow ~20 req/min (~333ms apart).
    /// Tier 3 endpoints allow ~50 req/min (~120ms apart).
    /// We use a conservative 200ms minimum interval for write endpoints.
    async fn rate_limit(&self, method: &str) {
        const MIN_INTERVAL: Duration = Duration::from_millis(200);

        if let Some(last) = self.rate_limits.get(method) {
            let elapsed = last.value().elapsed();
            if elapsed < MIN_INTERVAL {
                tokio::time::sleep(MIN_INTERVAL - elapsed).await;
            }
        }
        self.rate_limits.insert(method.to_string(), Instant::now());
    }

    /// [braingnosis] Resolve a platform_id (now user_id) to a Slack channel_id for API calls.
    ///
    /// Strategy:
    /// 1. If platform_id looks like a channel (starts with C/G), use it directly
    /// 2. Check `last_channel_for_user` cache (populated from incoming messages)
    /// 3. If it's a user ID (starts with U), open a DM via conversations.open
    /// 4. Fall back to using platform_id as-is (backward compat)
    async fn resolve_channel_id(&self, platform_id: &str) -> String {
        // Already a channel ID
        if platform_id.starts_with('C') || platform_id.starts_with('G') {
            return platform_id.to_string();
        }

        // Check last known channel for this user
        if let Some(channel) = self.last_channel_for_user.get(platform_id) {
            return channel.value().clone();
        }

        // User ID — open DM channel
        if platform_id.starts_with('U') {
            // Check DM channel cache
            if let Some(dm) = self.user_dm_channels.get(platform_id) {
                return dm.value().clone();
            }

            // Call conversations.open to get DM channel
            match self
                .client
                .post(format!("{SLACK_API_BASE}/conversations.open"))
                .header(
                    "Authorization",
                    format!("Bearer {}", self.bot_token.as_str()),
                )
                .json(&serde_json::json!({"users": platform_id}))
                .send()
                .await
            {
                Ok(resp) => {
                    if let Ok(json) = resp.json::<serde_json::Value>().await {
                        if json["ok"].as_bool() == Some(true) {
                            if let Some(ch) = json["channel"]["id"].as_str() {
                                self.user_dm_channels
                                    .insert(platform_id.to_string(), ch.to_string());
                                return ch.to_string();
                            }
                        } else {
                            warn!(
                                "Slack conversations.open failed: {}",
                                json["error"].as_str().unwrap_or("unknown")
                            );
                        }
                    }
                }
                Err(e) => {
                    warn!("Slack conversations.open HTTP error: {e}");
                }
            }
        }

        // Fallback
        platform_id.to_string()
    }

    /// [braingnosis] Send a reaction emoji to a Slack message via reactions.add.
    ///
    /// Slack reactions use shortcode names (e.g. "thinking_face"), not unicode emoji.
    async fn api_add_reaction(
        &self,
        channel_id: &str,
        timestamp: &str,
        emoji_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.rate_limit("reactions.add").await;
        let body = serde_json::json!({
            "channel": channel_id,
            "timestamp": timestamp,
            "name": emoji_name,
        });

        let resp: serde_json::Value = self
            .client
            .post(format!("{SLACK_API_BASE}/reactions.add"))
            .header(
                "Authorization",
                format!("Bearer {}", self.bot_token.as_str()),
            )
            .json(&body)
            .send()
            .await?
            .json()
            .await?;

        if resp["ok"].as_bool() != Some(true) {
            let err = resp["error"].as_str().unwrap_or("unknown");
            // "already_reacted" is not a real error — just means we already set this reaction.
            if err != "already_reacted" {
                warn!("Slack reactions.add failed: {err}");
            }
        }
        Ok(())
    }

    /// [braingnosis] Remove a reaction from a Slack message via reactions.remove.
    async fn api_remove_reaction(
        &self,
        channel_id: &str,
        timestamp: &str,
        emoji_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.rate_limit("reactions.remove").await;
        let body = serde_json::json!({
            "channel": channel_id,
            "timestamp": timestamp,
            "name": emoji_name,
        });

        let resp: serde_json::Value = self
            .client
            .post(format!("{SLACK_API_BASE}/reactions.remove"))
            .header(
                "Authorization",
                format!("Bearer {}", self.bot_token.as_str()),
            )
            .json(&body)
            .send()
            .await?
            .json()
            .await?;

        if resp["ok"].as_bool() != Some(true) {
            let err = resp["error"].as_str().unwrap_or("unknown");
            // "no_reaction" means it wasn't set — harmless.
            if err != "no_reaction" {
                warn!("Slack reactions.remove failed: {err}");
            }
        }
        Ok(())
    }

    /// Validate the bot token by calling auth.test.
    async fn validate_bot_token(&self) -> Result<String, Box<dyn std::error::Error>> {
        let resp: serde_json::Value = self
            .client
            .post(format!("{SLACK_API_BASE}/auth.test"))
            .header(
                "Authorization",
                format!("Bearer {}", self.bot_token.as_str()),
            )
            .send()
            .await?
            .json()
            .await?;

        if resp["ok"].as_bool() != Some(true) {
            let err = resp["error"].as_str().unwrap_or("unknown error");
            return Err(format!("Slack auth.test failed: {err}").into());
        }

        let user_id = resp["user_id"].as_str().unwrap_or("unknown").to_string();
        Ok(user_id)
    }

    /// Send a message to a Slack channel via chat.postMessage.
    async fn api_send_message(
        &self,
        channel_id: &str,
        text: &str,
        thread_ts: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // [braingnosis] Guard against empty text — Slack rejects with "no_text"
        if text.trim().is_empty() {
            tracing::debug!("Skipping empty Slack message to {channel_id}");
            return Ok(());
        }
        let chunks = split_message(text, SLACK_MSG_LIMIT);

        for chunk in chunks {
            self.rate_limit("chat.postMessage").await;
            let mut body = serde_json::json!({
                "channel": channel_id,
                "text": chunk,
                "unfurl_links": self.unfurl_links,
                "unfurl_media": self.unfurl_links,
            });
            if let Some(ts) = thread_ts {
                body["thread_ts"] = serde_json::json!(ts);
            }

            let resp: serde_json::Value = self
                .client
                .post(format!("{SLACK_API_BASE}/chat.postMessage"))
                .header(
                    "Authorization",
                    format!("Bearer {}", self.bot_token.as_str()),
                )
                .json(&body)
                .send()
                .await?
                .json()
                .await?;

            if resp["ok"].as_bool() != Some(true) {
                let err = resp["error"].as_str().unwrap_or("unknown");
                warn!("Slack chat.postMessage failed: {err}");
            }
        }
        Ok(())
    }

    /// [braingnosis] Edit an existing Slack message via chat.update.
    ///
    /// Used for live-updating streamed responses (edit the same message as tokens arrive).
    /// Returns the message timestamp for subsequent edits.
    #[allow(dead_code)] // Building block for future streaming edit support
    async fn api_update_message(
        &self,
        channel_id: &str,
        ts: &str,
        text: &str,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.rate_limit("chat.update").await;
        let body = serde_json::json!({
            "channel": channel_id,
            "ts": ts,
            "text": text,
        });

        let resp: serde_json::Value = self
            .client
            .post(format!("{SLACK_API_BASE}/chat.update"))
            .header(
                "Authorization",
                format!("Bearer {}", self.bot_token.as_str()),
            )
            .json(&body)
            .send()
            .await?
            .json()
            .await?;

        if resp["ok"].as_bool() != Some(true) {
            let err = resp["error"].as_str().unwrap_or("unknown");
            return Err(format!("Slack chat.update failed: {err}").into());
        }

        let ts = resp["ts"]
            .as_str()
            .unwrap_or(ts)
            .to_string();
        Ok(ts)
    }

    /// [braingnosis] Upload a file to Slack using the 3-step external upload flow.
    ///
    /// 1. `files.getUploadURLExternal` — get presigned upload URL + file_id
    /// 2. POST file bytes to the presigned URL
    /// 3. `files.completeUploadExternal` — finalize and share to channel
    async fn api_upload_file(
        &self,
        channel_id: &str,
        data: &[u8],
        filename: &str,
        thread_ts: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.rate_limit("files.upload").await;
        // Step 1: Get upload URL
        let params = vec![
            ("filename", filename.to_string()),
            ("length", data.len().to_string()),
        ];
        let resp: serde_json::Value = self
            .client
            .get(format!("{SLACK_API_BASE}/files.getUploadURLExternal"))
            .header(
                "Authorization",
                format!("Bearer {}", self.bot_token.as_str()),
            )
            .query(&params)
            .send()
            .await?
            .json()
            .await?;

        if resp["ok"].as_bool() != Some(true) {
            let err = resp["error"].as_str().unwrap_or("unknown");
            return Err(format!("Slack files.getUploadURLExternal failed: {err}").into());
        }

        let upload_url = resp["upload_url"]
            .as_str()
            .ok_or("Missing upload_url in response")?;
        let file_id = resp["file_id"]
            .as_str()
            .ok_or("Missing file_id in response")?;

        // Step 2: Upload file content to presigned URL
        let upload_resp = self
            .client
            .post(upload_url)
            .header("Content-Type", "application/octet-stream")
            .body(data.to_vec())
            .send()
            .await?;

        if !upload_resp.status().is_success() {
            return Err(format!(
                "Slack file upload to presigned URL failed: {}",
                upload_resp.status()
            )
            .into());
        }

        // Step 3: Complete upload and share to channel
        let mut complete_body = serde_json::json!({
            "files": [{"id": file_id, "title": filename}],
            "channel_id": channel_id,
        });
        if let Some(ts) = thread_ts {
            complete_body["thread_ts"] = serde_json::json!(ts);
        }

        let complete_resp: serde_json::Value = self
            .client
            .post(format!("{SLACK_API_BASE}/files.completeUploadExternal"))
            .header(
                "Authorization",
                format!("Bearer {}", self.bot_token.as_str()),
            )
            .json(&complete_body)
            .send()
            .await?
            .json()
            .await?;

        if complete_resp["ok"].as_bool() != Some(true) {
            let err = complete_resp["error"].as_str().unwrap_or("unknown");
            warn!("Slack files.completeUploadExternal failed: {err}");
        }

        Ok(())
    }
}

#[async_trait]
impl ChannelAdapter for SlackAdapter {
    fn name(&self) -> &str {
        "slack"
    }

    fn channel_type(&self) -> ChannelType {
        ChannelType::Slack
    }

    async fn start(
        &self,
    ) -> Result<Pin<Box<dyn Stream<Item = ChannelMessage> + Send>>, Box<dyn std::error::Error>>
    {
        // Validate bot token first
        let bot_user_id_val = self.validate_bot_token().await?;
        *self.bot_user_id.write().await = Some(bot_user_id_val.clone());
        info!("Slack bot authenticated (user_id: {bot_user_id_val})");

        let (tx, rx) = mpsc::channel::<ChannelMessage>(256);

        let app_token = self.app_token.clone();
        let bot_user_id = self.bot_user_id.clone();
        let allowed_channels = self.allowed_channels.clone();
        let client = self.client.clone();
        let mut shutdown = self.shutdown_rx.clone();
        let active_threads = self.active_threads.clone();
        let auto_thread_reply = self.auto_thread_reply;
        let last_channel_for_user = self.last_channel_for_user.clone();
        let user_names = self.user_names.clone();
        let bot_token = self.bot_token.clone();

        // Spawn periodic cleanup of expired thread entries.
        {
            let active_threads = self.active_threads.clone();
            let thread_ttl = self.thread_ttl;
            let mut cleanup_shutdown = self.shutdown_rx.clone();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(300));
                loop {
                    tokio::select! {
                        _ = interval.tick() => {
                            active_threads.retain(|_, last| last.elapsed() < thread_ttl);
                        }
                        _ = cleanup_shutdown.changed() => {
                            if *cleanup_shutdown.borrow() {
                                return;
                            }
                        }
                    }
                }
            });
        }

        tokio::spawn(async move {
            let mut backoff = INITIAL_BACKOFF;

            loop {
                if *shutdown.borrow() {
                    break;
                }

                // Get a fresh WebSocket URL
                let ws_url_result = get_socket_mode_url(&client, &app_token)
                    .await
                    .map_err(|e| e.to_string());
                let ws_url = match ws_url_result {
                    Ok(url) => url,
                    Err(err_msg) => {
                        warn!("Slack: failed to get WebSocket URL: {err_msg}, retrying in {backoff:?}");
                        tokio::time::sleep(backoff).await;
                        backoff = (backoff * 2).min(MAX_BACKOFF);
                        continue;
                    }
                };

                info!("Connecting to Slack Socket Mode...");

                let ws_result = tokio_tungstenite::connect_async(&ws_url).await;
                let ws_stream = match ws_result {
                    Ok((stream, _)) => stream,
                    Err(e) => {
                        warn!("Slack WebSocket connection failed: {e}, retrying in {backoff:?}");
                        tokio::time::sleep(backoff).await;
                        backoff = (backoff * 2).min(MAX_BACKOFF);
                        continue;
                    }
                };

                backoff = INITIAL_BACKOFF;
                info!("Slack Socket Mode connected");

                let (mut ws_tx, mut ws_rx) = ws_stream.split();

                let should_reconnect = 'inner: loop {
                    let msg = tokio::select! {
                        msg = ws_rx.next() => msg,
                        _ = shutdown.changed() => {
                            if *shutdown.borrow() {
                                let _ = ws_tx.close().await;
                                return;
                            }
                            continue;
                        }
                    };

                    let msg = match msg {
                        Some(Ok(m)) => m,
                        Some(Err(e)) => {
                            warn!("Slack WebSocket error: {e}");
                            break 'inner true;
                        }
                        None => {
                            info!("Slack WebSocket closed");
                            break 'inner true;
                        }
                    };

                    let text = match msg {
                        tokio_tungstenite::tungstenite::Message::Text(t) => t,
                        tokio_tungstenite::tungstenite::Message::Close(_) => {
                            info!("Slack Socket Mode closed by server");
                            break 'inner true;
                        }
                        _ => continue,
                    };

                    let payload: serde_json::Value = match serde_json::from_str(&text) {
                        Ok(v) => v,
                        Err(e) => {
                            warn!("Slack: failed to parse message: {e}");
                            continue;
                        }
                    };

                    let envelope_type = payload["type"].as_str().unwrap_or("");

                    match envelope_type {
                        "hello" => {
                            debug!("Slack Socket Mode hello received");
                        }

                        "events_api" => {
                            // Acknowledge the envelope
                            let envelope_id = payload["envelope_id"].as_str().unwrap_or("");
                            if !envelope_id.is_empty() {
                                let ack = serde_json::json!({ "envelope_id": envelope_id });
                                if let Err(e) = ws_tx
                                    .send(tokio_tungstenite::tungstenite::Message::Text(
                                        serde_json::to_string(&ack).unwrap(),
                                    ))
                                    .await
                                {
                                    error!("Slack: failed to send ack: {e}");
                                    break 'inner true;
                                }
                            }

                            // Extract the event
                            let event = &payload["payload"]["event"];
                            if let Some(mut msg) = parse_slack_event(
                                event,
                                &bot_user_id,
                                &allowed_channels,
                                &active_threads,
                                auto_thread_reply,
                                Some(&last_channel_for_user),
                            )
                            .await
                            {
                                // [braingnosis] Resolve user display name via users.info API
                                let display_name = resolve_user_name(
                                    &client,
                                    &bot_token,
                                    &msg.sender.platform_id,
                                    &user_names,
                                )
                                .await;
                                msg.sender.display_name = display_name;

                                debug!(
                                    "Slack message from {}: {:?}",
                                    msg.sender.display_name, msg.content
                                );
                                if tx.send(msg).await.is_err() {
                                    return;
                                }
                            }
                        }

                        "disconnect" => {
                            let reason = payload["reason"].as_str().unwrap_or("unknown");
                            info!("Slack disconnect request: {reason}");
                            break 'inner true;
                        }

                        _ => {
                            debug!("Slack envelope type: {envelope_type}");
                        }
                    }
                };

                if !should_reconnect || *shutdown.borrow() {
                    break;
                }

                warn!("Slack: reconnecting in {backoff:?}");
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(MAX_BACKOFF);
            }

            info!("Slack Socket Mode loop stopped");
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Box::pin(stream))
    }

    async fn send(
        &self,
        user: &ChannelUser,
        content: ChannelContent,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // [braingnosis] platform_id is now user_id (for RBAC), resolve to channel_id for API
        let channel_id = self.resolve_channel_id(&user.platform_id).await;
        match content {
            ChannelContent::Text(text) => {
                self.api_send_message(&channel_id, &text, None).await?;
            }
            // [braingnosis] File upload support via 3-step external upload flow
            ChannelContent::FileData {
                data,
                filename,
                mime_type: _,
            } => {
                self.api_upload_file(&channel_id, &data, &filename, None)
                    .await?;
            }
            ChannelContent::File { url, filename } => {
                // Send as a link — the URL is already accessible
                let text = format!("[{filename}]({url})");
                self.api_send_message(&channel_id, &text, None).await?;
            }
            _ => {
                self.api_send_message(&channel_id, "(Unsupported content type)", None)
                    .await?;
            }
        }
        Ok(())
    }

    async fn send_in_thread(
        &self,
        user: &ChannelUser,
        content: ChannelContent,
        thread_id: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // [braingnosis] platform_id is now user_id (for RBAC), resolve to channel_id for API
        let channel_id = self.resolve_channel_id(&user.platform_id).await;

        // [braingnosis] DM channels (starting with 'D') don't support threading in Slack.
        // Replies with thread_ts in a DM create a confusing sub-thread under each message.
        // Instead, send as a flat reply so the conversation reads naturally.
        let thread_ts = if channel_id.starts_with('D') {
            None
        } else {
            Some(thread_id)
        };

        match content {
            ChannelContent::Text(text) => {
                self.api_send_message(&channel_id, &text, thread_ts)
                    .await?;
            }
            // [braingnosis] File upload support via 3-step external upload flow
            ChannelContent::FileData {
                data,
                filename,
                mime_type: _,
            } => {
                self.api_upload_file(&channel_id, &data, &filename, thread_ts)
                    .await?;
            }
            ChannelContent::File { url, filename } => {
                let text = format!("[{filename}]({url})");
                self.api_send_message(&channel_id, &text, thread_ts)
                    .await?;
            }
            _ => {
                self.api_send_message(&channel_id, "(Unsupported content type)", thread_ts)
                    .await?;
            }
        }
        Ok(())
    }

    /// [braingnosis] Send a lifecycle reaction (emoji) to a Slack message.
    ///
    /// Uses the Slack `reactions.add` API. Optionally removes the previous phase reaction
    /// when `reaction.remove_previous` is true.
    async fn send_reaction(
        &self,
        user: &ChannelUser,
        message_id: &str,
        reaction: &LifecycleReaction,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let channel_id = self.resolve_channel_id(&user.platform_id).await;
        let reaction_key = format!("{channel_id}:{message_id}");

        // Map unicode emoji to Slack shortcode
        let shortcode = match emoji_to_slack_shortcode(&reaction.emoji) {
            Some(sc) => sc,
            None => {
                debug!(
                    "No Slack shortcode for emoji '{}', skipping reaction",
                    reaction.emoji
                );
                return Ok(());
            }
        };

        // Remove previous reaction if requested
        if reaction.remove_previous {
            if let Some(prev) = self.last_reaction.get(&reaction_key) {
                let prev_code = prev.value().clone();
                if prev_code != shortcode {
                    let _ = self
                        .api_remove_reaction(&channel_id, message_id, &prev_code)
                        .await;
                }
            }
        }

        // Add the new reaction
        self.api_add_reaction(&channel_id, message_id, shortcode)
            .await?;

        // Track for future remove_previous
        self.last_reaction
            .insert(reaction_key, shortcode.to_string());

        Ok(())
    }

    async fn stop(&self) -> Result<(), Box<dyn std::error::Error>> {
        let _ = self.shutdown_tx.send(true);
        Ok(())
    }
}

/// [braingnosis] Map unicode emoji to Slack reaction shortcode name.
///
/// Slack's reactions.add API requires the text name (e.g. "thinking_face"),
/// not the unicode codepoint. This maps the lifecycle emoji used by OpenFang.
fn emoji_to_slack_shortcode(emoji: &str) -> Option<&'static str> {
    match emoji {
        "\u{1F914}" => Some("thinking_face"),                   // 🤔
        "\u{2699}\u{FE0F}" | "\u{2699}" => Some("gear"),       // ⚙️
        "\u{270D}\u{FE0F}" | "\u{270D}" => Some("writing_hand"), // ✍️
        "\u{2705}" => Some("white_check_mark"),                 // ✅
        "\u{274C}" => Some("x"),                                // ❌
        "\u{23F3}" => Some("hourglass_flowing_sand"),           // ⏳
        "\u{1F504}" => Some("arrows_counterclockwise"),         // 🔄
        "\u{1F440}" => Some("eyes"),                            // 👀
        _ => None,
    }
}

/// Helper to get Socket Mode WebSocket URL.
async fn get_socket_mode_url(
    client: &reqwest::Client,
    app_token: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let resp: serde_json::Value = client
        .post(format!("{SLACK_API_BASE}/apps.connections.open"))
        .header("Authorization", format!("Bearer {app_token}"))
        .header("Content-Type", "application/x-www-form-urlencoded")
        .send()
        .await?
        .json()
        .await?;

    if resp["ok"].as_bool() != Some(true) {
        let err = resp["error"].as_str().unwrap_or("unknown error");
        return Err(format!("Slack apps.connections.open failed: {err}").into());
    }

    resp["url"]
        .as_str()
        .map(String::from)
        .ok_or_else(|| "Missing 'url' in connections.open response".into())
}

/// [braingnosis] Resolve a Slack user_id to a display name via users.info API.
///
/// Free function for use inside spawned tasks that don't have `&self`.
/// Checks the DashMap cache first, then calls the API and caches the result.
async fn resolve_user_name(
    client: &reqwest::Client,
    bot_token: &str,
    user_id: &str,
    cache: &DashMap<String, String>,
) -> String {
    // Check cache first
    if let Some(name) = cache.get(user_id) {
        return name.value().clone();
    }

    // Call users.info API
    let result = client
        .get(format!("{SLACK_API_BASE}/users.info?user={user_id}"))
        .header("Authorization", format!("Bearer {bot_token}"))
        .send()
        .await;

    match result {
        Ok(resp) => {
            if let Ok(json) = resp.json::<serde_json::Value>().await {
                if json["ok"].as_bool() == Some(true) {
                    let user_obj = &json["user"];
                    // Prefer display_name > real_name > name > user_id
                    let display_name = user_obj["profile"]["display_name"]
                        .as_str()
                        .filter(|s| !s.is_empty())
                        .or_else(|| {
                            user_obj["profile"]["real_name"]
                                .as_str()
                                .filter(|s| !s.is_empty())
                        })
                        .or_else(|| user_obj["real_name"].as_str().filter(|s| !s.is_empty()))
                        .or_else(|| user_obj["name"].as_str().filter(|s| !s.is_empty()))
                        .unwrap_or(user_id)
                        .to_string();

                    cache.insert(user_id.to_string(), display_name.clone());
                    return display_name;
                } else {
                    let err = json["error"].as_str().unwrap_or("unknown");
                    warn!("Slack users.info failed for {user_id}: {err}");
                }
            }
        }
        Err(e) => {
            warn!("Slack users.info HTTP error for {user_id}: {e}");
        }
    }

    // Fallback: cache user_id so we don't keep retrying failed lookups
    cache.insert(user_id.to_string(), user_id.to_string());
    user_id.to_string()
}

/// Parse a Slack event into a `ChannelMessage`.
async fn parse_slack_event(
    event: &serde_json::Value,
    bot_user_id: &Arc<RwLock<Option<String>>>,
    allowed_channels: &[String],
    active_threads: &Arc<DashMap<String, Instant>>,
    auto_thread_reply: bool,
    last_channel_for_user: Option<&Arc<DashMap<String, String>>>, // [braingnosis] Track user→channel mapping
) -> Option<ChannelMessage> {
    let event_type = event["type"].as_str()?;
    if event_type != "message" && event_type != "app_mention" {
        return None;
    }

    // Handle message_changed subtype: extract inner message
    let subtype = event["subtype"].as_str();
    let (msg_data, is_edit) = match subtype {
        Some("message_changed") => {
            // Edited messages have the new content in event.message
            match event.get("message") {
                Some(inner) => (inner, true),
                None => return None,
            }
        }
        Some(_) => return None, // Skip other subtypes (joins, leaves, etc.)
        None => (event, false),
    };

    // Filter out bot's own messages
    if msg_data.get("bot_id").is_some() {
        return None;
    }
    let user_id = msg_data["user"]
        .as_str()
        .or_else(|| event["user"].as_str())?;
    if let Some(ref bid) = *bot_user_id.read().await {
        if user_id == bid {
            return None;
        }
    }

    let channel = event["channel"].as_str()?;

    // Filter by allowed channels
    if !allowed_channels.is_empty() && !allowed_channels.contains(&channel.to_string()) {
        return None;
    }

    let text = msg_data["text"].as_str().unwrap_or("");

    // [braingnosis] Check for file attachments in the message
    let has_files = msg_data["files"].is_array()
        && !msg_data["files"].as_array().map_or(true, |a| a.is_empty());

    // Allow messages that have text OR files (not both empty)
    if text.is_empty() && !has_files {
        return None;
    }

    let ts = if is_edit {
        msg_data["ts"]
            .as_str()
            .unwrap_or(event["ts"].as_str().unwrap_or("0"))
    } else {
        event["ts"].as_str().unwrap_or("0")
    };

    // Parse timestamp (Slack uses epoch.microseconds format)
    let timestamp = ts
        .split('.')
        .next()
        .and_then(|s| s.parse::<i64>().ok())
        .and_then(|epoch| chrono::DateTime::from_timestamp(epoch, 0))
        .unwrap_or_else(chrono::Utc::now);

    // [braingnosis] Parse content: prioritize files if present, otherwise text/commands
    let content = if has_files {
        // Extract first file — Slack provides url_private for bot-accessible download
        let file = &msg_data["files"][0];
        let filename = file["name"]
            .as_str()
            .unwrap_or("attachment")
            .to_string();
        let url = file["url_private"]
            .as_str()
            .or_else(|| file["url_private_download"].as_str())
            .unwrap_or("")
            .to_string();
        if url.is_empty() {
            // File without URL — fall back to text
            if text.is_empty() {
                return None;
            }
            ChannelContent::Text(text.to_string())
        } else {
            // If there's also text, prepend it as context
            if !text.is_empty() {
                // We can only return one content type, so embed the text in metadata later.
                // For now, return the file content — the text is usually the file caption.
                ChannelContent::File { url, filename }
            } else {
                ChannelContent::File { url, filename }
            }
        }
    } else if text.starts_with('/') {
        let parts: Vec<&str> = text.splitn(2, ' ').collect();
        let cmd_name = &parts[0][1..];
        let args = if parts.len() > 1 {
            parts[1].split_whitespace().map(String::from).collect()
        } else {
            vec![]
        };
        ChannelContent::Command {
            name: cmd_name.to_string(),
            args,
        }
    } else {
        ChannelContent::Text(text.to_string())
    };

    // Extract thread_id: threaded replies have `thread_ts`, top-level messages
    // use their own `ts` so the reply will start a thread under the original.
    // [braingnosis] DM flat-reply handling is done in send_in_thread() instead —
    // the adapter strips thread_ts for DM channels (starts with 'D').
    let real_thread_ts = msg_data["thread_ts"]
        .as_str()
        .or_else(|| event["thread_ts"].as_str());
    let channel_type_raw = event["channel_type"].as_str().unwrap_or("");
    let is_dm = channel_type_raw == "im" || channel.starts_with('D');
    let thread_id = real_thread_ts
        .map(|s| s.to_string())
        .or_else(|| Some(ts.to_string()));

    // Check if the bot was @-mentioned (for group_policy = "mention_only")
    let mut metadata = HashMap::new();
    if event_type == "app_mention" {
        metadata.insert("was_mentioned".to_string(), serde_json::Value::Bool(true));
    }

    // Determine the real thread_ts from the event (None for top-level messages).

    let mut explicitly_mentioned = false;
    if let Some(ref bid) = *bot_user_id.read().await {
        let mention_tag = format!("<@{bid}>");
        if text.contains(&mention_tag) {
            explicitly_mentioned = true;
            metadata.insert("was_mentioned".to_string(), serde_json::json!(true));

            // Track thread for auto-reply on subsequent messages.
            if let Some(tts) = real_thread_ts {
                active_threads.insert(tts.to_string(), Instant::now());
            }
        }
    }

    // Auto-reply to follow-up messages in tracked threads.
    if !explicitly_mentioned && auto_thread_reply {
        if let Some(tts) = real_thread_ts {
            if let Some(mut entry) = active_threads.get_mut(tts) {
                // Refresh TTL and mark as mentioned so dispatch proceeds.
                *entry = Instant::now();
                metadata.insert("was_mentioned".to_string(), serde_json::json!(true));
            }
        }
    }

    // [braingnosis] is_group based on channel type, not hardcoded.
    let is_group = !is_dm;

    // [braingnosis] Store channel ID in metadata so send() can route replies correctly.
    // platform_id is now user_id (for RBAC), but we still need the channel for API calls.
    metadata.insert(
        "slack_channel_id".to_string(),
        serde_json::Value::String(channel.to_string()),
    );

    // [braingnosis] Track user_id → channel_id for reply routing
    if let Some(cache) = last_channel_for_user {
        cache.insert(user_id.to_string(), channel.to_string());
    }

    Some(ChannelMessage {
        channel: ChannelType::Slack,
        platform_message_id: ts.to_string(),
        sender: ChannelUser {
            platform_id: user_id.to_string(), // [braingnosis] Fixed: was channel.to_string() — broke RBAC user matching
            display_name: user_id.to_string(), // TODO: resolve via users.info API (Step 7)
            openfang_user: None,
        },
        content,
        target_agent: None,
        timestamp,
        is_group, // [braingnosis] Fixed: was hardcoded `true` — broke DM policy
        thread_id,
        metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_parse_slack_event_basic() {
        let bot_id = Arc::new(RwLock::new(Some("B123".to_string())));
        let event = serde_json::json!({
            "type": "message",
            "user": "U456",
            "channel": "C789",
            "text": "Hello agent!",
            "ts": "1700000000.000100"
        });

        let msg = parse_slack_event(&event, &bot_id, &[], &Arc::new(DashMap::new()), true, None)
            .await
            .unwrap();
        assert_eq!(msg.channel, ChannelType::Slack);
        assert_eq!(msg.sender.platform_id, "U456"); // [braingnosis] Fixed: now user_id, not channel
        assert!(matches!(msg.content, ChannelContent::Text(ref t) if t == "Hello agent!"));
    }

    #[tokio::test]
    async fn test_parse_slack_event_filters_bot() {
        let bot_id = Arc::new(RwLock::new(Some("B123".to_string())));
        let event = serde_json::json!({
            "type": "message",
            "user": "U456",
            "channel": "C789",
            "text": "Bot message",
            "ts": "1700000000.000100",
            "bot_id": "B999"
        });

        let msg = parse_slack_event(&event, &bot_id, &[], &Arc::new(DashMap::new()), true, None).await;
        assert!(msg.is_none());
    }

    #[tokio::test]
    async fn test_parse_slack_event_filters_own_user() {
        let bot_id = Arc::new(RwLock::new(Some("U456".to_string())));
        let event = serde_json::json!({
            "type": "message",
            "user": "U456",
            "channel": "C789",
            "text": "My message",
            "ts": "1700000000.000100"
        });

        let msg = parse_slack_event(&event, &bot_id, &[], &Arc::new(DashMap::new()), true, None).await;
        assert!(msg.is_none());
    }

    #[tokio::test]
    async fn test_parse_slack_event_channel_filter() {
        let bot_id = Arc::new(RwLock::new(None));
        let event = serde_json::json!({
            "type": "message",
            "user": "U456",
            "channel": "C789",
            "text": "Hello",
            "ts": "1700000000.000100"
        });

        // Not in allowed channels
        let msg = parse_slack_event(
            &event,
            &bot_id,
            &["C111".to_string(), "C222".to_string()],
            &Arc::new(DashMap::new()),
            true,
            None,
        )
        .await;
        assert!(msg.is_none());

        // In allowed channels
        let msg = parse_slack_event(
            &event,
            &bot_id,
            &["C789".to_string()],
            &Arc::new(DashMap::new()),
            true,
            None,
        )
        .await;
        assert!(msg.is_some());
    }

    #[tokio::test]
    async fn test_parse_slack_event_skips_other_subtypes() {
        let bot_id = Arc::new(RwLock::new(None));
        // Non-message_changed subtypes should still be filtered
        let event = serde_json::json!({
            "type": "message",
            "subtype": "channel_join",
            "user": "U456",
            "channel": "C789",
            "text": "joined",
            "ts": "1700000000.000100"
        });

        let msg = parse_slack_event(&event, &bot_id, &[], &Arc::new(DashMap::new()), true, None).await;
        assert!(msg.is_none());
    }

    #[tokio::test]
    async fn test_parse_slack_command() {
        let bot_id = Arc::new(RwLock::new(None));
        let event = serde_json::json!({
            "type": "message",
            "user": "U456",
            "channel": "C789",
            "text": "/agent hello-world",
            "ts": "1700000000.000100"
        });

        let msg = parse_slack_event(&event, &bot_id, &[], &Arc::new(DashMap::new()), true, None)
            .await
            .unwrap();
        match &msg.content {
            ChannelContent::Command { name, args } => {
                assert_eq!(name, "agent");
                assert_eq!(args, &["hello-world"]);
            }
            other => panic!("Expected Command, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_parse_slack_event_message_changed() {
        let bot_id = Arc::new(RwLock::new(Some("B123".to_string())));
        let event = serde_json::json!({
            "type": "message",
            "subtype": "message_changed",
            "channel": "C789",
            "message": {
                "user": "U456",
                "text": "Edited message text",
                "ts": "1700000000.000100"
            },
            "ts": "1700000001.000200"
        });

        let msg = parse_slack_event(&event, &bot_id, &[], &Arc::new(DashMap::new()), true, None)
            .await
            .unwrap();
        assert_eq!(msg.channel, ChannelType::Slack);
        assert_eq!(msg.sender.platform_id, "U456"); // [braingnosis] Fixed: now user_id, not channel
        assert!(matches!(msg.content, ChannelContent::Text(ref t) if t == "Edited message text"));
    }

    // [braingnosis] New tests for RBAC fix
    #[tokio::test]
    async fn test_parse_slack_event_dm_detection() {
        let bot_id = Arc::new(RwLock::new(Some("B123".to_string())));
        // DM channels start with "D" and have channel_type "im"
        let dm_event = serde_json::json!({
            "type": "message",
            "user": "U456",
            "channel": "D789ABC",
            "channel_type": "im",
            "text": "Hello in DM",
            "ts": "1700000000.000100"
        });
        let msg = parse_slack_event(&dm_event, &bot_id, &[], &Arc::new(DashMap::new()), true, None)
            .await
            .unwrap();
        assert!(!msg.is_group, "DM should have is_group=false");

        // Channel messages should be is_group=true
        let channel_event = serde_json::json!({
            "type": "message",
            "user": "U456",
            "channel": "C789",
            "channel_type": "channel",
            "text": "Hello in channel",
            "ts": "1700000000.000100"
        });
        let msg = parse_slack_event(&channel_event, &bot_id, &[], &Arc::new(DashMap::new()), true, None)
            .await
            .unwrap();
        assert!(msg.is_group, "Channel should have is_group=true");
    }

    #[tokio::test]
    async fn test_parse_slack_event_dm_detection_by_channel_prefix() {
        let bot_id = Arc::new(RwLock::new(Some("B123".to_string())));
        // Even without channel_type field, DM channels detected by "D" prefix
        let dm_event = serde_json::json!({
            "type": "message",
            "user": "U456",
            "channel": "D789ABC",
            "text": "Hello in DM without channel_type",
            "ts": "1700000000.000100"
        });
        let msg = parse_slack_event(&dm_event, &bot_id, &[], &Arc::new(DashMap::new()), true, None)
            .await
            .unwrap();
        assert!(!msg.is_group, "D-prefixed channel should be detected as DM");
    }

    #[tokio::test]
    async fn test_parse_slack_event_user_id_in_platform_id() {
        let bot_id = Arc::new(RwLock::new(None));
        let event = serde_json::json!({
            "type": "message",
            "user": "U999TESTUSER",
            "channel": "C789",
            "text": "Test RBAC",
            "ts": "1700000000.000100"
        });
        let msg = parse_slack_event(&event, &bot_id, &[], &Arc::new(DashMap::new()), true, None)
            .await
            .unwrap();
        assert_eq!(msg.sender.platform_id, "U999TESTUSER", "platform_id must be user_id for RBAC");
        assert_ne!(msg.sender.platform_id, "C789", "platform_id must NOT be channel_id");
    }

    #[test]
    fn test_slack_adapter_creation() {
        let adapter = SlackAdapter::new(
            "xapp-test".to_string(),
            "xoxb-test".to_string(),
            vec!["C123".to_string()],
            true,
            24,
            true,
        );
        assert_eq!(adapter.name(), "slack");
        assert_eq!(adapter.channel_type(), ChannelType::Slack);
    }

    #[test]
    fn test_slack_adapter_unfurl_links_enabled() {
        let adapter = SlackAdapter::new(
            "xapp-test".to_string(),
            "xoxb-test".to_string(),
            vec![],
            true,
            24,
            true,
        );
        assert!(adapter.unfurl_links);
    }

    #[test]
    fn test_slack_adapter_unfurl_links_disabled() {
        let adapter = SlackAdapter::new(
            "xapp-test".to_string(),
            "xoxb-test".to_string(),
            vec![],
            true,
            24,
            false,
        );
        assert!(!adapter.unfurl_links);
    }

    // [braingnosis] Reaction emoji shortcode mapping tests
    #[test]
    fn test_emoji_to_slack_shortcode() {
        assert_eq!(emoji_to_slack_shortcode("\u{1F914}"), Some("thinking_face"));
        assert_eq!(emoji_to_slack_shortcode("\u{2699}\u{FE0F}"), Some("gear"));
        assert_eq!(emoji_to_slack_shortcode("\u{2699}"), Some("gear")); // without variation selector
        assert_eq!(emoji_to_slack_shortcode("\u{2705}"), Some("white_check_mark"));
        assert_eq!(emoji_to_slack_shortcode("\u{274C}"), Some("x"));
        assert_eq!(emoji_to_slack_shortcode("\u{23F3}"), Some("hourglass_flowing_sand"));
        assert_eq!(emoji_to_slack_shortcode("\u{1F504}"), Some("arrows_counterclockwise"));
        assert_eq!(emoji_to_slack_shortcode("\u{1F440}"), Some("eyes"));
        assert_eq!(emoji_to_slack_shortcode("unknown"), None);
    }

    #[test]
    fn test_emoji_shortcode_covers_all_lifecycle_phases() {
        use crate::types::{default_phase_emoji, AgentPhase};
        // Every phase's default emoji should map to a Slack shortcode
        let phases = [
            AgentPhase::Queued,
            AgentPhase::Thinking,
            AgentPhase::ToolUse { tool_name: "test".into() },
            AgentPhase::Streaming,
            AgentPhase::Done,
            AgentPhase::Error,
        ];
        for phase in &phases {
            let emoji = default_phase_emoji(phase);
            assert!(
                emoji_to_slack_shortcode(emoji).is_some(),
                "No Slack shortcode for phase {phase:?} emoji '{emoji}'"
            );
        }
    }

    // [braingnosis] File message parsing tests
    #[tokio::test]
    async fn test_parse_slack_event_with_file() {
        let bot_id = Arc::new(RwLock::new(Some("B123".to_string())));
        let event = serde_json::json!({
            "type": "message",
            "user": "U456",
            "channel": "C789",
            "text": "",
            "ts": "1700000000.000100",
            "files": [{
                "id": "F1234",
                "name": "report.pdf",
                "url_private": "https://files.slack.com/files-pri/T123-F1234/report.pdf",
                "mimetype": "application/pdf",
                "size": 12345
            }]
        });

        let msg = parse_slack_event(&event, &bot_id, &[], &Arc::new(DashMap::new()), true, None)
            .await
            .unwrap();
        match &msg.content {
            ChannelContent::File { url, filename } => {
                assert_eq!(filename, "report.pdf");
                assert!(url.contains("files.slack.com"));
            }
            other => panic!("Expected File content, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_parse_slack_event_file_only_no_text() {
        let bot_id = Arc::new(RwLock::new(None));
        // Message with file but no text should still be parsed
        let event = serde_json::json!({
            "type": "message",
            "user": "U456",
            "channel": "C789",
            "text": "",
            "ts": "1700000000.000100",
            "files": [{
                "id": "F5678",
                "name": "image.png",
                "url_private": "https://files.slack.com/files-pri/T123-F5678/image.png"
            }]
        });

        let msg = parse_slack_event(&event, &bot_id, &[], &Arc::new(DashMap::new()), true, None)
            .await;
        assert!(msg.is_some(), "File-only message should be parsed");
    }
}

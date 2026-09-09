/**
 * Client-side HTTP client for inference.
 * Routes directly to provider APIs using user-configured endpoints and API keys.
 * Handles streaming responses in-browser.
 */

export interface InferenceResult {
  thoughtTokens: string;
  actionJson: string;
  rtt: number;
  inferenceTime: number;
}

export interface ConnectionTestResult {
  ok: boolean;
  latencyMs: number;
  error?: string;
}

import { PromptAssembler } from './PromptAssembler';

export class InferenceClient {
  private providerType: string = 'kaggle';
  private endpoint: string = '';
  private apiKey?: string = '';
  private model: string = '';

  // Circuit breaker: exponential backoff on connection failures
  private backoffUntil = 0;
  private backoffMs = 5000;

  /** True when the client is in backoff after consecutive connection failures. */
  public isInBackoff(): boolean {
    return Date.now() < this.backoffUntil;
  }

  public setProvider(type: string, endpoint: string, apiKey?: string, model?: string) {
    this.providerType = (type || '').trim();
    this.endpoint = (endpoint || '').trim();
    this.apiKey = apiKey ? apiKey.trim() : undefined;
    this.model = model ? model.trim() : '';
    this.backoffUntil = 0;
    this.backoffMs = 5000;
    console.log(`[InferenceClient] Set provider client-side: type=${this.providerType}, endpoint=${this.endpoint}, model=${this.model}`);
  }

  public hasEndpoint(): boolean {
    return !!this.endpoint;
  }

  /**
   * True only when the provider has an endpoint/key sufficient to run a real cycle.
   * Used by AgentLoop to avoid cycling (and flickering status) when nothing is connected.
   */
  public hasCredentials(): boolean {
    if (this.providerType === 'kaggle' || this.providerType === 'ollama' || this.providerType === 'lmstudio') {
      return !!this.endpoint;
    }
    return !!this.endpoint && !!this.apiKey;
  }

  /** Build the direct API URL for any provider. */
  private getDirectUrl(): string {
    const clean = this.endpoint.trim().replace(/\/+$/, '');
    if (this.providerType === 'kaggle') {
      if (clean.endsWith('/infer') || clean.endsWith('/chat/completions')) {
        return clean;
      }
      return `${clean}/infer`;
    }
    return clean.endsWith('/chat/completions')
      ? clean
      : `${clean}/chat/completions`;
  }

  private buildOpenAIMessages(payload: any): any[] {
    const isLocal = this.providerType === 'kaggle'
                 || this.providerType === 'ollama'
                 || this.providerType === 'lmstudio';
    const assembled = isLocal
      ? PromptAssembler.build(payload)
      : PromptAssembler.buildCompact(payload);
    const systemText = assembled.systemPrompt;

    const userParts: any[] = [];

    // 1. If Video Task is active, attach demonstration target frame first with clear labeling
    if (payload.video_task?.target_frame) {
      const targetUrl = payload.video_task.target_frame.startsWith('data:')
        ? payload.video_task.target_frame
        : `data:image/webp;base64,${payload.video_task.target_frame}`;
      userParts.push({
        type: 'text',
        text: `[REFERENCE DEMONSTRATION FRAME - Milestone ${payload.video_task.milestone_index} of ${payload.video_task.total_milestones}: ${payload.video_task.label}]`
      });
      userParts.push({ type: 'image_url', image_url: { url: targetUrl } });
      userParts.push({
        type: 'text',
        text: `[YOUR CURRENT FIRST-PERSON LIVE VIEW]`
      });
    }

    if (payload.frame) {
      const imageUrl = payload.frame.startsWith('data:')
        ? payload.frame
        : `data:image/webp;base64,${payload.frame}`;
      userParts.push({ type: 'image_url', image_url: { url: imageUrl } });
    }

    const jointsText = payload._delta_joints
      ? JSON.stringify(payload._delta_joints)
      : JSON.stringify(payload.joints);
    const directiveLine = payload._directive_text || '';
    const tactile = payload.tactile_context || 'No tactile data.';
    userParts.push({
      type: 'text',
      text: `${directiveLine}Joints: ${jointsText}.\nTactile: ${tactile}`
    });

    const perception = payload.perception_summary || '';
    if (perception) {
      userParts.push({ type: 'text', text: `\nSPATIAL GROUNDING:\n${perception}` });
    }

    const codex = payload.motor_codex_hints;
    if (codex) {
      userParts.push({ type: 'text', text: `\n${codex}` });
    }

    const physicalFeedback = payload.physical_feedback;
    if (physicalFeedback) {
      userParts.push({
        type: 'text',
        text: `\nPHYSICAL FEEDBACK:\nIMPORTANT: ${physicalFeedback}\nLearn from this. Your body has real physical limits.`
      });
    }

    const identityFeedback = payload.identity_feedback;
    if (identityFeedback) {
      userParts.push({
        type: 'text',
        text: `\nIDENTITY FEEDBACK:\n${identityFeedback}\nCorrect your identity_update format and try again in the next cycle.`
      });
    }

    userParts.push({
      type: 'text',
      text: `\nENVIRONMENTAL AWARENESS:\nSometimes your visual field may appear as pure darkness. Use joint data when the image is uninformative. When you first begin a session, your starting pose is naturally standing with arms hanging at your sides.`
    });

    const injection = payload.pending_injection;
    if (injection) {
      userParts.push({
        type: 'text',
        text: `\n🚨 USER OVERRIDE DIRECTIVE 🚨\nYou MUST obey the following injected instruction immediately: ${injection}\nAcknowledge this directive in your thought stream.`
      });
    }

    return [
      { role: 'system', content: systemText },
      { role: 'user', content: userParts }
    ];
  }

  /**
   * Minimal connectivity test against the configured provider.
   * Sends a tiny text request directly to the provider API, without streaming.
   * Returns a pass/fail result plus measured latency.
   */
  public async testConnection(): Promise<ConnectionTestResult> {
    const startTime = Date.now();

    if (this.providerType === 'kaggle') {
      const clean = this.endpoint.trim().replace(/\/+$/, '').replace(/\/infer$/, '').replace(/\/chat\/completions$/, '');
      const healthUrl = `${clean}/health`;
      try {
        const response = await fetch(healthUrl, {
          method: 'GET',
          signal: AbortSignal.timeout(10000),
        });
        const latencyMs = Date.now() - startTime;
        if (!response.ok) {
          return { ok: false, latencyMs, error: `HTTP ${response.status}` };
        }
        return { ok: true, latencyMs };
      } catch (err: any) {
        return {
          ok: false,
          latencyMs: Date.now() - startTime,
          error: err?.message || String(err),
        };
      }
    }

    const url = this.getDirectUrl();
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    const systemText = 'Respond with exactly: OK';
    const userText = 'Connectivity test. Reply with only the word OK.';

    const body = {
      model: this.model || 'default',
      messages: [
        { role: 'system', content: systemText },
        { role: 'user', content: userText },
      ],
      stream: false,
      max_tokens: 10,
      temperature: 0,
    };

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(15000),
      });

      const latencyMs = Date.now() - startTime;

      if (!response.ok) {
        let detail = '';
        try {
          const errText = await response.text();
          try {
            const errJson = JSON.parse(errText);
            detail = errJson.error || errJson.details || JSON.stringify(errJson).slice(0, 300);
          } catch {
            detail = errText.slice(0, 300);
          }
        } catch {
          detail = `HTTP ${response.status} (no body available)`;
        }
        return { ok: false, latencyMs, error: `HTTP ${response.status}: ${detail}` };
      }

      return { ok: true, latencyMs: Date.now() - startTime };
    } catch (err: any) {
      return {
        ok: false,
        latencyMs: Date.now() - startTime,
        error: err?.message || String(err),
      };
    }
  }

  public async infer(payload: any, onToken: (token: string) => void): Promise<InferenceResult> {
    // Circuit breaker: reject immediately during backoff to avoid zombie HTTP spam
    if (Date.now() < this.backoffUntil) {
      throw new Error(`InferenceClient: backoff active, retry after ${this.backoffUntil}`);
    }

    const startTime = Date.now();
    let firstTokenTime = 0;

    const url = this.getDirectUrl();
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    // For kaggle, POST the raw InferPayload to /infer (plain-text streaming).
    // For all other providers, use OpenAI chat-completions format.
    const body = this.providerType === 'kaggle'
      ? payload
      : {
          model: this.model || 'default',
          messages: this.buildOpenAIMessages(payload),
          stream: true,
          max_tokens: 4096,
          temperature: 0.7,
          top_p: 0.9,
        };

    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(60000),
    }).catch((err: any) => {
      // Connection error (ERR_CONNECTION_REFUSED, NetworkError, etc.) — trigger backoff
      const isConnectionError = err?.message?.includes('Failed to fetch') ||
                                err?.message?.includes('NetworkError') ||
                                err?.message?.includes('ERR_CONNECTION_REFUSED') ||
                                err?.name === 'TypeError' ||
                                err?.name === 'TimeoutError';
      if (isConnectionError) {
        this.backoffUntil = Date.now() + this.backoffMs;
        this.backoffMs = Math.min(this.backoffMs * 2, 60000);
        console.warn(`[InferenceClient] Connection failed or timed out, backing off ${this.backoffMs}ms`);
      }
      throw err;
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`Inference HTTP error ${response.status}: ${errText}`);
    }

    console.log(`[InferenceClient] Connected to ${url} (status ${response.status}), reading stream...`);

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Response body has no reader (streaming unsupported by browser)');
    }

    const contentType = response.headers.get('content-type') || '';
    const isSSE = contentType.includes('text/event-stream');

    const decoder = new TextDecoder('utf-8');
    let lineBuffer = '';
    let buffer = '';
    let thoughtTokens = '';
    let actionJson = '';
    let isAction = false;
    const separator = '---ACTION---';

    const processDelta = (delta: string) => {
      if (!delta) return;

      if (!isAction) {
        buffer += delta;
        const idx = buffer.indexOf(separator);
        if (idx !== -1) {
          const thoughtPart = buffer.substring(0, idx);
          const newThought = thoughtPart.substring(thoughtTokens.length);
          if (newThought) onToken(newThought);
          thoughtTokens = thoughtPart;
          isAction = true;
          actionJson = buffer.substring(idx + separator.length);
        } else {
          // Stream thought tokens safely before separator
          const safeLen = buffer.length - separator.length + 1;
          if (safeLen > thoughtTokens.length) {
            const newThought = buffer.substring(thoughtTokens.length, safeLen);
            onToken(newThought);
            thoughtTokens = buffer.substring(0, safeLen);
          }
        }
      } else {
        actionJson += delta;
      }
    };

    // Check if the response body starts with or contains SSE data: markers or if content-type declares it
    let detectedSSE = isSSE;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        if (firstTokenTime === 0) firstTokenTime = Date.now();

        const chunkText = decoder.decode(value, { stream: true });

        if (!detectedSSE && chunkText.includes('data: ')) {
          detectedSSE = true;
        }

        // If server returns raw text (non-SSE), stream chunk directly
        if (!detectedSSE) {
          processDelta(chunkText);
          continue;
        }

        // SSE line processing
        lineBuffer += chunkText;
        const lines = lineBuffer.split('\n');
        lineBuffer = lines.pop() || ''; // Keep trailing partial line in buffer

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || trimmed === 'data: [DONE]' || trimmed === '[DONE]') continue;

          let payloadStr = '';
          if (trimmed.startsWith('data: ')) {
            payloadStr = trimmed.slice(6).trim();
          } else if (trimmed.startsWith('{')) {
            payloadStr = trimmed;
          } else {
            // Raw text line fallback
            processDelta(line + '\n');
            continue;
          }

          if (payloadStr === '[DONE]' || !payloadStr) continue;

          try {
            const parsed = JSON.parse(payloadStr);
            const delta = parsed.choices?.[0]?.delta?.content 
                       ?? parsed.choices?.[0]?.message?.content
                       ?? parsed.text
                       ?? parsed.response
                       ?? '';
            if (delta) {
              processDelta(delta);
            }
          } catch {
            // Non-JSON SSE string fallback
            processDelta(payloadStr);
          }
        }
      }

      // Process any leftover line in lineBuffer
      if (lineBuffer.trim()) {
        const trimmed = lineBuffer.trim();
        if (trimmed && trimmed !== '[DONE]' && trimmed !== 'data: [DONE]') {
          const payloadStr = trimmed.startsWith('data: ') ? trimmed.slice(6).trim() : trimmed;
          try {
            const parsed = JSON.parse(payloadStr);
            const delta = parsed.choices?.[0]?.delta?.content || '';
            if (delta) processDelta(delta);
          } catch {
            processDelta(payloadStr);
          }
        }
      }
    } finally {
      reader.releaseLock();
    }

    if (!isAction) {
      const jsonStart = buffer.indexOf('{');
      if (jsonStart !== -1) {
        thoughtTokens = buffer.substring(0, jsonStart);
        actionJson = buffer.substring(jsonStart);
      } else {
        thoughtTokens = buffer;
      }
      const remainingThought = thoughtTokens.substring(thoughtTokens.length);
      if (remainingThought) onToken(remainingThought);
    }

    const endTime = Date.now();

    // Success — reset backoff
    this.backoffMs = 5000;
    this.backoffUntil = 0;

    return {
      thoughtTokens,
      actionJson,
      rtt: firstTokenTime - startTime,
      inferenceTime: endTime - firstTokenTime,
    };
  }
}

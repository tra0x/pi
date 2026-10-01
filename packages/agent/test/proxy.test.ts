import type {
	Api,
	AssistantMessage,
	AssistantMessageEvent,
	Message,
	Model,
	SimpleStreamOptions,
	TranscriptContext,
} from "@earendil-works/pi-ai";
import { normalizeContext } from "@earendil-works/pi-ai";
import { anthropicMessagesApi, getModel } from "@earendil-works/pi-ai/compat";
import { afterEach, describe, expect, it, vi } from "vitest";
import { type ProxyAssistantMessageEvent, streamProxy } from "../src/proxy.ts";

const model: Model<"openai-responses"> = {
	id: "gpt-5.4",
	name: "GPT-5.4",
	api: "openai-responses",
	provider: "openai",
	baseUrl: "https://api.openai.com/v1",
	reasoning: true,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 400000,
	maxTokens: 128000,
};

const usage: AssistantMessage["usage"] = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

afterEach(() => {
	vi.unstubAllGlobals();
});

describe("streamProxy", () => {
	it("preserves tool-call metadata received only on toolcall_end", async () => {
		const proxyEvents: ProxyAssistantMessageEvent[] = [
			{ type: "start" },
			{ type: "toolcall_start", contentIndex: 0, id: "call_test|fc_test", toolName: "lookup" },
			{ type: "toolcall_delta", contentIndex: 0, delta: '{"value":"hello"}' },
			{
				type: "toolcall_end",
				contentIndex: 0,
				toolCall: {
					type: "toolCall",
					id: "call_test|fc_test",
					name: "lookup",
					arguments: { value: "hello" },
					namespace: "dynamic_tools",
				},
			},
			{ type: "done", reason: "toolUse", usage },
		];
		const body = proxyEvents.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("");
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => new Response(body, { status: 200 })),
		);

		const stream = streamProxy(model, normalizeContext({ systemPrompt: "", messages: [] }), {
			authToken: "test-token",
			proxyUrl: "https://proxy.example.com",
		});
		const events: AssistantMessageEvent[] = [];
		for await (const event of stream) events.push(event);
		const result = await stream.result();
		const endEvent = events.find((event) => event.type === "toolcall_end");

		expect(endEvent).toMatchObject({
			type: "toolcall_end",
			toolCall: { namespace: "dynamic_tools" },
		});
		expect(result.content[0]).toMatchObject({
			type: "toolCall",
			arguments: { value: "hello" },
			namespace: "dynamic_tools",
		});
	});

	// Regression tests for https://github.com/earendil-works/pi/issues/8996
	it("processes terminal metadata when the event is not newline-terminated", async () => {
		const start = `data: ${JSON.stringify({ type: "start" })}\n\n`;
		const done = `data: ${JSON.stringify({ type: "done", reason: "stop", usage, providerThinkingLevel: "high" })}`;
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => new Response(start + done, { status: 200 })),
		);

		const stream = streamProxy(model, normalizeContext({ systemPrompt: "", messages: [] }), {
			authToken: "test-token",
			proxyUrl: "https://proxy.example.com",
		});
		const events: AssistantMessageEvent[] = [];
		for await (const event of stream) events.push(event);
		const result = await stream.result();

		expect(events.map((event) => event.type)).toEqual(["start", "done"]);
		expect(result.stopReason).toBe("stop");
		expect(result.providerThinkingLevel).toBe("high");
	});

	it("emits an error instead of hanging when the stream ends without a terminal event", async () => {
		const body = `data: ${JSON.stringify({ type: "start" })}\n\n`;
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => new Response(body, { status: 200 })),
		);

		const stream = streamProxy(model, normalizeContext({ systemPrompt: "", messages: [] }), {
			authToken: "test-token",
			proxyUrl: "https://proxy.example.com",
		});
		const events: AssistantMessageEvent[] = [];
		for await (const event of stream) events.push(event);
		const result = await stream.result();

		expect(events.map((event) => event.type)).toEqual(["start", "error"]);
		expect(result.stopReason).toBe("error");
		expect(result.errorMessage).toContain("Connection closed by proxy server");
	});
});

// Regression tests for https://github.com/earendil-works/pi/issues/9074. The server below is a
// pi-ai-based backend with a test-local serializer that, like ProxyAssistantMessageEvent, drops
// `partial` and terminal message metadata such as `anthropicFallbacks`.
describe("streamProxy with an Anthropic fallback after output", () => {
	type ProxyRequestBody = { model: Model<Api>; context: TranscriptContext; options: SimpleStreamOptions };
	type AnthropicPayload = { messages: Array<{ role: string; content: unknown }> };

	const anthropicModel = getModel("anthropic", "claude-opus-5");
	const anthropic = anthropicMessagesApi();
	const userHello: Message = { role: "user", content: "Hello", timestamp: 1 };

	function anthropicSse(events: Array<{ type: string } & Record<string, unknown>>): Response {
		const body = events.map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join("");
		return new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } });
	}

	function block(index: number, contentBlock: Record<string, unknown>, deltas: Record<string, unknown>[] = []) {
		return [
			{ type: "content_block_start", index, content_block: contentBlock },
			...deltas.map((delta) => ({ type: "content_block_delta", index, delta })),
			{ type: "content_block_stop", index },
		];
	}

	const midOutputFallback = () =>
		anthropicSse([
			{
				type: "message_start",
				message: { id: "msg_a", model: "claude-opus-5", usage: { input_tokens: 10, output_tokens: 0 } },
			},
			...block(0, { type: "thinking", thinking: "", signature: "" }, [
				{ type: "thinking_delta", thinking: "declined reasoning" },
				{ type: "signature_delta", signature: "sig-a" },
			]),
			...block(1, { type: "text", text: "" }, [{ type: "text_delta", text: "partial" }]),
			...block(2, { type: "fallback", from: { model: "claude-opus-5" }, to: { model: "claude-opus-4-8" } }),
			...block(3, { type: "thinking", thinking: "", signature: "" }, [
				{ type: "thinking_delta", thinking: "fallback reasoning" },
				{ type: "signature_delta", signature: "sig-b" },
			]),
			...block(4, { type: "text", text: "" }, [{ type: "text_delta", text: "fallback answer" }]),
			{ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 5 } },
			{ type: "message_stop" },
		]);

	const plainAnswer = () =>
		anthropicSse([
			{
				type: "message_start",
				message: { id: "msg_b", model: "claude-opus-5", usage: { input_tokens: 10, output_tokens: 0 } },
			},
			...block(0, { type: "text", text: "" }, [{ type: "text_delta", text: "ok" }]),
			{ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 1 } },
			{ type: "message_stop" },
		]);

	function toProxyEvent(event: AssistantMessageEvent): ProxyAssistantMessageEvent {
		switch (event.type) {
			case "start":
				return { type: "start" };
			case "text_start":
			case "thinking_start":
				return { type: event.type, contentIndex: event.contentIndex };
			case "text_delta":
			case "thinking_delta":
			case "toolcall_delta":
				return { type: event.type, contentIndex: event.contentIndex, delta: event.delta };
			case "text_end": {
				const content = event.partial.content[event.contentIndex];
				const contentSignature = content?.type === "text" ? content.textSignature : undefined;
				return { type: "text_end", contentIndex: event.contentIndex, contentSignature };
			}
			case "thinking_end": {
				const content = event.partial.content[event.contentIndex];
				const contentSignature = content?.type === "thinking" ? content.thinkingSignature : undefined;
				return { type: "thinking_end", contentIndex: event.contentIndex, contentSignature };
			}
			case "toolcall_start": {
				const content = event.partial.content[event.contentIndex];
				if (content?.type !== "toolCall") throw new Error("Expected a tool call");
				return { type: "toolcall_start", contentIndex: event.contentIndex, id: content.id, toolName: content.name };
			}
			case "toolcall_end":
				return { type: "toolcall_end", contentIndex: event.contentIndex, toolCall: event.toolCall };
			case "done":
				if (event.reason === "deferred") throw new Error("Deferred responses are not proxied");
				return { type: "done", reason: event.reason, usage: event.message.usage };
			case "error":
				return {
					type: "error",
					reason: event.reason,
					errorMessage: event.error.errorMessage,
					usage: event.error.usage,
				};
		}
	}

	/** Runs one proxied turn followed by a second turn, returning what reached Anthropic. */
	async function runConversation(serverRetainsFinalMessage: boolean) {
		const proxyRequests: ProxyRequestBody[] = [];
		const anthropicPayloads: AnthropicPayload[] = [];
		const anthropicResponses = [midOutputFallback(), plainAnswer()];
		const upstream: typeof fetch = async (_input, init) => {
			anthropicPayloads.push(JSON.parse(String(init?.body)) as AnthropicPayload);
			const response = anthropicResponses.shift();
			if (!response) throw new Error("Unexpected Anthropic request");
			return response;
		};
		vi.stubGlobal(
			"fetch",
			vi.fn(async (_input: string, init: RequestInit) => {
				const request = JSON.parse(String(init.body)) as ProxyRequestBody;
				proxyRequests.push(request);
				const events: ProxyAssistantMessageEvent[] = [];
				for await (const event of anthropic.streamSimple(request.model, request.context, {
					...request.options,
					apiKey: "server-key",
					cacheRetention: "none",
					fetch: upstream,
					// Only for the contrast case: this server still strips metadata.
					...(serverRetainsFinalMessage ? { retainsFinalMessage: true } : {}),
				})) {
					events.push(toProxyEvent(event));
				}
				return new Response(events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""), {
					status: 200,
				});
			}),
		);
		// Callers spread their own options into streamProxy, as in the README example.
		const callerOptions: SimpleStreamOptions = { retainsFinalMessage: true, maxTokens: 100 };
		const proxyOptions = { ...callerOptions, authToken: "test-token", proxyUrl: "https://proxy.example.com" };

		const first = await streamProxy(
			anthropicModel,
			normalizeContext({ messages: [userHello] }),
			proxyOptions,
		).result();
		const next: Message = { role: "user", content: "Next", timestamp: 3 };
		await streamProxy(
			anthropicModel,
			normalizeContext({ messages: [userHello, first, next] }),
			proxyOptions,
		).result();
		return { first, proxyRequests, anthropicPayloads };
	}

	function assistantBlocks(payload: AnthropicPayload): Array<Record<string, unknown>> {
		return payload.messages
			.filter((message) => message.role === "assistant")
			.flatMap((message) =>
				Array.isArray(message.content) ? (message.content as Array<Record<string, unknown>>) : [],
			);
	}

	it("fails the turn and keeps the next request free of declined output", async () => {
		const { first, proxyRequests, anthropicPayloads } = await runConversation(false);

		expect(first.stopReason).toBe("error");
		expect(first.errorMessage).toBe("Anthropic performed an unsupported mid-output model fallback");
		for (const request of proxyRequests) {
			expect(request.options).toEqual({ maxTokens: 100 });
			expect(request.options).not.toHaveProperty("retainsFinalMessage");
		}
		expect(anthropicPayloads).toHaveLength(2);
		expect(assistantBlocks(anthropicPayloads[1])).toEqual([]);
		expect(JSON.stringify(anthropicPayloads[1])).not.toContain("sig-a");
	});

	it("replays declined signed thinking without its marker when a stripping server opts in", async () => {
		// Shows why only callers that keep the final message may set retainsFinalMessage.
		const { first, anthropicPayloads } = await runConversation(true);

		expect(first.stopReason).toBe("stop");
		expect(first.anthropicFallbacks).toBeUndefined();
		const replayed = assistantBlocks(anthropicPayloads[1]);
		expect(replayed).toContainEqual({ type: "thinking", thinking: "declined reasoning", signature: "sig-a" });
		expect(replayed.some((block) => block.type === "fallback")).toBe(false);
	});
});

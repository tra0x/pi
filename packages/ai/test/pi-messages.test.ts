import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, it } from "vitest";
import { streamSimple as anthropicStreamSimple } from "../src/api/anthropic-messages.ts";
import { type PiMessagesEvent, type PiMessagesOptions, stream, streamSimple } from "../src/api/pi-messages.ts";
import { getModel } from "../src/compat.ts";
import type {
	Api,
	AssistantMessageEvent,
	Context,
	Message,
	Model,
	StopReason,
	TranscriptContext,
} from "../src/types.ts";
import { normalizeContext } from "../src/utils/transcript.ts";

type RecordedRequest = {
	url: string;
	headers: IncomingMessage["headers"];
	body: unknown;
};

type ResponderOptions = {
	status?: number;
	headers?: Record<string, string>;
	events?: unknown[];
	rawBody?: string;
};

let server: Server | undefined;

afterEach(() => {
	server?.close();
	server = undefined;
});

async function startServer(options: ResponderOptions): Promise<{ baseUrl: string; requests: RecordedRequest[] }> {
	const requests: RecordedRequest[] = [];

	server = createServer((request: IncomingMessage, response: ServerResponse) => {
		const chunks: Buffer[] = [];
		request.on("data", (chunk: Buffer) => chunks.push(chunk));
		request.on("end", () => {
			const raw = Buffer.concat(chunks).toString("utf-8");
			requests.push({
				url: request.url ?? "",
				headers: request.headers,
				body: raw ? JSON.parse(raw) : undefined,
			});

			if (options.status && options.status !== 200) {
				response.statusCode = options.status;
				response.setHeader("content-type", "application/json");
				response.end(options.rawBody ?? "{}");
				return;
			}

			response.statusCode = 200;
			response.setHeader("content-type", "text/event-stream");
			for (const [name, value] of Object.entries(options.headers ?? {})) {
				response.setHeader(name, value);
			}
			for (const event of options.events ?? []) {
				response.write(`data: ${JSON.stringify(event)}\n\n`);
			}
			response.end();
		});
	});

	await new Promise<void>((resolve) => {
		server!.listen(0, "127.0.0.1", () => resolve());
	});

	const address = server!.address() as AddressInfo;
	return { baseUrl: `http://127.0.0.1:${address.port}/v1`, requests };
}

function createModel(baseUrl: string): Model<"pi-messages"> {
	return {
		id: "auto",
		name: "Radius Auto",
		api: "pi-messages",
		provider: "radius",
		baseUrl,
		reasoning: false,
		input: ["text"],
		cost: { input: 1, output: 2, cacheRead: 0.1, cacheWrite: 0.2 },
		contextWindow: 128000,
		maxTokens: 16384,
	};
}

const context: Context = {
	messages: [{ role: "user", content: "Hello", timestamp: Date.now() }],
};

const usage = {
	input: 10,
	output: 5,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 15,
	cost: { input: 0.1, output: 0.2, cacheRead: 0, cacheWrite: 0, total: 0.3 },
};

describe("pi-messages", () => {
	it("streams text and tool calls and resolves the terminal message", async () => {
		const { baseUrl, requests } = await startServer({
			events: [
				{ type: "start" },
				{ type: "text_start", contentIndex: 0 },
				{ type: "text_delta", contentIndex: 0, delta: "Hel" },
				{ type: "text_delta", contentIndex: 0, delta: "lo" },
				{ type: "text_end", contentIndex: 0, content: "Hello" },
				{ type: "toolcall_start", contentIndex: 1, id: "call_1", toolName: "read" },
				{ type: "toolcall_delta", contentIndex: 1, delta: '{"path":' },
				{ type: "toolcall_delta", contentIndex: 1, delta: '"a.txt"}' },
				{
					type: "toolcall_end",
					contentIndex: 1,
					toolCall: { type: "toolCall", id: "call_1", name: "read", arguments: { path: "a.txt" } },
				},
				{
					type: "done",
					reason: "toolUse",
					usage,
					responseId: "resp_1",
					providerThinkingLevel: "high",
				},
			],
		});
		const model = createModel(baseUrl);

		const events: AssistantMessageEvent[] = [];
		const partialStopReasons: StopReason[] = [];
		const eventStream = stream(model, normalizeContext(context), {
			apiKey: "test-key",
			sessionId: "session-1",
			toolChoice: "auto",
			maxTokens: 100,
			headers: { "x-custom": "1" },
		});
		for await (const event of eventStream) {
			if ("partial" in event) {
				partialStopReasons.push(event.partial.stopReason);
			}
			events.push(event);
		}
		const message = await eventStream.result();

		expect(partialStopReasons[0]).toBe("pending");
		expect(message.stopReason).toBe("toolUse");
		expect(message.usage).toEqual(usage);
		expect(message.responseId).toBe("resp_1");
		expect(message.providerThinkingLevel).toBe("high");
		expect(message.model).toBe("auto");
		expect(message.provider).toBe("radius");
		expect(message.content).toEqual([
			{ type: "text", text: "Hello", textSignature: undefined },
			{ type: "toolCall", id: "call_1", name: "read", arguments: { path: "a.txt" } },
		]);
		expect(events.some((event) => event.type === "text_delta")).toBe(true);
		expect(events.filter((event) => event.type === "toolcall_end")).toHaveLength(1);

		expect(requests).toHaveLength(1);
		const request = requests[0];
		expect(request.url).toBe("/v1/messages");
		expect(request.headers.authorization).toBe("Bearer test-key");
		expect(request.headers["x-custom"]).toBe("1");
		expect(request.body).toEqual({
			model: "auto",
			context,
			options: { maxTokens: 100, sessionId: "session-1", toolChoice: "auto" },
		});
	});

	it("forwards parsed wire events in order before converting them", async () => {
		const wireEvents = [
			{ type: "start" },
			{ type: "text_start", contentIndex: 0 },
			{ type: "text_delta", contentIndex: 0, delta: "Hello", gatewayField: "upstream-value" },
			{ type: "text_end", contentIndex: 0, content: "Hello" },
			{ type: "done", reason: "stop", usage, responseId: "resp_1" },
		];
		const { baseUrl } = await startServer({ events: wireEvents });
		const model = createModel(baseUrl);
		const received: unknown[] = [];
		const eventModels: Model<Api>[] = [];
		const message = await streamSimple(model, normalizeContext(context), {
			apiKey: "test-key",
			onProviderStreamEvent: async (event, eventModel) => {
				await Promise.resolve();
				received.push(event);
				eventModels.push(eventModel);
			},
		}).result();

		expect(received).toEqual(wireEvents);
		expect(eventModels).toEqual(wireEvents.map(() => model));
		expect(message.stopReason).toBe("stop");
		expect(message.responseId).toBe("resp_1");
		expect(message.content).toEqual([{ type: "text", text: "Hello", textSignature: undefined }]);
	});

	it("appends debug=1 and reports response headers via onResponse", async () => {
		const { baseUrl, requests } = await startServer({
			headers: { "x-pi-gateway-upstream-provider": "anthropic" },
			events: [{ type: "done", reason: "stop", usage }],
		});
		const model = createModel(baseUrl);

		let observedHeaders: Record<string, string> | undefined;
		const options = {
			apiKey: "test-key",
			debug: true,
			onResponse: (response) => {
				observedHeaders = response.headers;
			},
		} satisfies PiMessagesOptions;
		const message = await streamSimple(model, normalizeContext(context), options).result();

		expect(message.stopReason).toBe("stop");
		expect(requests[0].url).toBe("/v1/messages?debug=1");
		expect(observedHeaders?.["x-pi-gateway-upstream-provider"]).toBe("anthropic");
	});

	it("surfaces backend error responses with diagnostics", async () => {
		const { baseUrl } = await startServer({
			status: 401,
			rawBody: JSON.stringify({ error: { message: "Token expired", code: "unauthorized" } }),
		});
		const model = createModel(baseUrl);

		const message = await stream(model, normalizeContext(context), { apiKey: "stale" }).result();

		expect(message.stopReason).toBe("error");
		expect(message.errorMessage).toContain("401");
		expect(message.errorMessage).toContain("Token expired");
		expect(message.errorMessage).toContain("unauthorized");
		expect(message.diagnostics?.[0]?.type).toBe("pi_messages_response_failure");
		expect(message.diagnostics?.[0]?.details?.status).toBe(401);
	});

	it("propagates server-sent error events", async () => {
		const { baseUrl } = await startServer({
			events: [{ type: "start" }, { type: "error", reason: "error", usage, errorMessage: "Upstream failed" }],
		});
		const model = createModel(baseUrl);

		const message = await stream(model, normalizeContext(context), { apiKey: "test-key" }).result();

		expect(message.stopReason).toBe("error");
		expect(message.errorMessage).toBe("Upstream failed");
		expect(message.usage).toEqual(usage);
	});

	it("errors when no API key is provided", async () => {
		const model = createModel("http://127.0.0.1:1/v1");

		const message = await stream(model, normalizeContext(context)).result();

		expect(message.stopReason).toBe("error");
		expect(message.errorMessage).toContain("No API key provided");
	});

	it("errors when the stream ends without a terminal event", async () => {
		const { baseUrl } = await startServer({
			events: [
				{ type: "start" },
				{ type: "text_start", contentIndex: 0 },
				{ type: "text_delta", contentIndex: 0, delta: "partial" },
			],
		});
		const model = createModel(baseUrl);

		const message = await stream(model, normalizeContext(context), { apiKey: "test-key" }).result();

		expect(message.stopReason).toBe("error");
		expect(message.errorMessage).toContain("stream ended without a terminal event");
	});
});

describe("pi-messages api registration", () => {
	it("is registered as a builtin api provider", async () => {
		const { getApiProvider } = await import("../src/compat.ts");
		expect(getApiProvider("pi-messages")).toBeDefined();
	});

	it("is a known api usable on models", () => {
		const api: Api = "pi-messages";
		expect(api).toBe("pi-messages");
	});
});

// Regression tests for https://github.com/earendil-works/pi/issues/9074. The backend below runs
// pi-ai's Anthropic adapter and serializes PiMessagesEvent, which drops `partial` and terminal
// message metadata such as `anthropicFallbacks`.
describe("pi-messages with an Anthropic fallback after output", () => {
	type BackendRequest = { model: string; context: TranscriptContext; options: Record<string, unknown> };
	type AnthropicPayload = { messages: Array<{ role: string; content: unknown }> };

	const anthropicModel = getModel("anthropic", "claude-opus-5");

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
			...block(3, { type: "text", text: "" }, [{ type: "text_delta", text: "fallback answer" }]),
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

	function toPiMessagesEvent(event: AssistantMessageEvent): PiMessagesEvent {
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
				return { type: "text_end", contentIndex: event.contentIndex, content: event.content, contentSignature };
			}
			case "thinking_end": {
				const content = event.partial.content[event.contentIndex];
				const thinking = content?.type === "thinking" ? content : undefined;
				return {
					type: "thinking_end",
					contentIndex: event.contentIndex,
					content: event.content,
					contentSignature: thinking?.thinkingSignature,
					redacted: thinking?.redacted,
				};
			}
			case "toolcall_start": {
				const content = event.partial.content[event.contentIndex];
				if (content?.type !== "toolCall") throw new Error("Expected a tool call");
				return { type: "toolcall_start", contentIndex: event.contentIndex, id: content.id, toolName: content.name };
			}
			case "toolcall_end":
				return { type: "toolcall_end", contentIndex: event.contentIndex, toolCall: event.toolCall };
			case "done":
				if (event.reason === "deferred") throw new Error("Deferred responses are not served");
				return {
					type: "done",
					reason: event.reason,
					usage: event.message.usage,
					responseId: event.message.responseId,
				};
			case "error":
				return {
					type: "error",
					reason: event.reason,
					usage: event.error.usage,
					errorMessage: event.error.errorMessage,
					responseId: event.error.responseId,
				};
		}
	}

	/**
	 * Runs two client turns through a backend. With `relabel`, the backend rewrites history to the
	 * underlying Anthropic identity, so same-model signed thinking would be replayed.
	 */
	async function runConversation(relabel: boolean) {
		const backendRequests: BackendRequest[] = [];
		const anthropicPayloads: AnthropicPayload[] = [];
		const anthropicResponses = [midOutputFallback(), plainAnswer()];
		const upstream: typeof fetch = async (_input, init) => {
			anthropicPayloads.push(JSON.parse(String(init?.body)) as AnthropicPayload);
			const response = anthropicResponses.shift();
			if (!response) throw new Error("Unexpected Anthropic request");
			return response;
		};
		const backend: typeof fetch = async (_input, init) => {
			const request = JSON.parse(String(init?.body)) as BackendRequest;
			backendRequests.push(request);
			const messages = relabel
				? request.context.messages.map((message) =>
						message.role === "assistant"
							? {
									...message,
									api: anthropicModel.api,
									provider: anthropicModel.provider,
									model: anthropicModel.id,
								}
							: message,
					)
				: request.context.messages;
			const events: PiMessagesEvent[] = [];
			for await (const event of anthropicStreamSimple(
				anthropicModel,
				{ ...request.context, messages },
				{ apiKey: "server-key", cacheRetention: "none", fetch: upstream },
			)) {
				events.push(toPiMessagesEvent(event));
			}
			return new Response(events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""), {
				status: 200,
				headers: { "content-type": "text/event-stream" },
			});
		};
		const model = createModel("https://backend.example.com/v1");
		const userHello: Message = { role: "user", content: "Hello", timestamp: 1 };
		const options = { apiKey: "test-key", fetch: backend, retainsFinalMessage: true };

		const first = await stream(model, normalizeContext({ messages: [userHello] }), options).result();
		const next: Message = { role: "user", content: "Next", timestamp: 3 };
		await stream(model, normalizeContext({ messages: [userHello, first, next] }), options).result();
		return { first, backendRequests, anthropicPayloads };
	}

	for (const relabel of [false, true]) {
		it(`fails the turn and keeps the next request clean with ${relabel ? "relabelled" : "as-sent"} identity`, async () => {
			const { first, backendRequests, anthropicPayloads } = await runConversation(relabel);

			expect(first.stopReason).toBe("error");
			expect(first.errorMessage).toBe("Anthropic performed an unsupported mid-output model fallback");
			expect(first.model).toBe("auto");
			for (const request of backendRequests) {
				expect(request.options).not.toHaveProperty("retainsFinalMessage");
			}
			// The client sends its history raw, including the errored turn; the backend drops it.
			expect(backendRequests[1].context.messages.map((message) => message.role)).toEqual([
				"user",
				"assistant",
				"user",
			]);
			expect(anthropicPayloads).toHaveLength(2);
			expect(anthropicPayloads[1].messages.map((message) => message.role)).not.toContain("assistant");
			expect(JSON.stringify(anthropicPayloads[1])).not.toContain("sig-a");
			expect(JSON.stringify(anthropicPayloads[1])).not.toContain("partial");
		});
	}
});

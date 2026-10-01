import type Anthropic from "@anthropic-ai/sdk";
import { Type } from "typebox";
import { describe, expect, it } from "vitest";
import {
	type AnthropicOptions,
	stream as streamAnthropic,
	streamSimple as streamAnthropicSimple,
} from "../src/api/anthropic-messages.ts";
import { transformMessages } from "../src/api/transform-messages.ts";
import { getModel, normalizeContext } from "../src/compat.ts";
import type { Api, AssistantMessage, AssistantMessageEvent, Message, Model, ToolCall } from "../src/types.ts";

function createSseResponse(events: Array<{ event: string; data: string }>): Response {
	const body = events.map(({ event, data }) => `event: ${event}\ndata: ${data}\n`).join("\n");
	return new Response(body, {
		status: 200,
		headers: { "content-type": "text/event-stream" },
	});
}

const minimalAnthropicEvents = [
	{
		event: "message_start",
		data: JSON.stringify({
			type: "message_start",
			message: {
				id: "msg_test",
				usage: {
					input_tokens: 12,
					output_tokens: 0,
					cache_read_input_tokens: 0,
					cache_creation_input_tokens: 0,
				},
			},
		}),
	},
	{
		event: "content_block_start",
		data: JSON.stringify({
			type: "content_block_start",
			index: 0,
			content_block: { type: "text", text: "" },
		}),
	},
	{
		event: "content_block_delta",
		data: JSON.stringify({
			type: "content_block_delta",
			index: 0,
			delta: { type: "text_delta", text: "Hello" },
		}),
	},
	{
		event: "content_block_stop",
		data: JSON.stringify({ type: "content_block_stop", index: 0 }),
	},
	{
		event: "message_delta",
		data: JSON.stringify({
			type: "message_delta",
			delta: { stop_reason: "end_turn" },
			usage: {
				input_tokens: 12,
				output_tokens: 5,
				cache_read_input_tokens: 0,
				cache_creation_input_tokens: 0,
			},
		}),
	},
	{
		event: "message_stop",
		data: JSON.stringify({ type: "message_stop" }),
	},
];

function createFakeAnthropicClient(response: Response): Anthropic {
	return {
		beta: {
			messages: {
				create: () => ({
					asResponse: async () => response,
				}),
			},
		},
	} as unknown as Anthropic;
}

type ResponseContentBlock = { type: "thinking"; thinking: string; signature: string } | { type: "text"; text: string };

function createResponseModelSseResponse(model: string, contentBlock: ResponseContentBlock): Response {
	return createSseResponse([
		{
			event: "message_start",
			data: JSON.stringify({
				type: "message_start",
				message: { id: "msg_response_model", model, usage: { input_tokens: 100, output_tokens: 0 } },
			}),
		},
		{
			event: "content_block_start",
			data: JSON.stringify({ type: "content_block_start", index: 0, content_block: contentBlock }),
		},
		{ event: "content_block_stop", data: JSON.stringify({ type: "content_block_stop", index: 0 }) },
		{
			event: "message_delta",
			data: JSON.stringify({
				type: "message_delta",
				delta: { stop_reason: "end_turn" },
				usage: { input_tokens: 100, output_tokens: 20 },
			}),
		},
		{ event: "message_stop", data: JSON.stringify({ type: "message_stop" }) },
	]);
}

type StreamEventData = { type: string } & Record<string, unknown>;

function createEventResponse(events: StreamEventData[]): Response {
	return createSseResponse(events.map((data) => ({ event: data.type, data: JSON.stringify(data) })));
}

function blockEvents(
	index: number,
	contentBlock: Record<string, unknown>,
	deltas: Record<string, unknown>[] = [],
): StreamEventData[] {
	return [
		{ type: "content_block_start", index, content_block: contentBlock },
		...deltas.map((delta) => ({ type: "content_block_delta", index, delta })),
		{ type: "content_block_stop", index },
	];
}

type CapturedPayload = {
	messages: Array<{ role: string; content: string | Array<Record<string, unknown>> }>;
	fallbacks?: unknown;
	betas?: string[];
};

function createCapturingAnthropicClient(payloads: CapturedPayload[]): Anthropic {
	return {
		beta: {
			messages: {
				create: (params: CapturedPayload) => {
					payloads.push(params);
					return { asResponse: async () => createSseResponse(minimalAnthropicEvents) };
				},
			},
		},
	} as unknown as Anthropic;
}

async function captureReplayPayload(
	model: Model<"anthropic-messages">,
	messages: Message[],
	options?: AnthropicOptions,
): Promise<CapturedPayload> {
	const payloads: CapturedPayload[] = [];
	await streamAnthropic(model, normalizeContext({ messages }), {
		...options,
		client: createCapturingAnthropicClient(payloads),
	}).result();
	return payloads[0];
}

function assistantPayloadContent(payload: CapturedPayload): unknown[] {
	return payload.messages.filter((message) => message.role === "assistant").map((message) => message.content);
}

function withFallbackModels(
	model: Model<"anthropic-messages">,
	fallbacks: Array<{ model: string; input: number; output: number }>,
): Model<"anthropic-messages"> {
	return {
		...model,
		compat: {
			...model.compat,
			allowedFallbackModels: fallbacks.map((fallback) => ({
				provider: model.provider,
				model: fallback.model,
				cost: { input: fallback.input, output: fallback.output, cacheRead: 0, cacheWrite: 0 },
			})),
		},
	};
}

const userHello: Message = { role: "user", content: "Hello", timestamp: 1 };

const fallbackToOpus48 = {
	type: "fallback",
	from: { model: "claude-opus-5" },
	to: { model: "claude-opus-4-8" },
	trigger: { type: "refusal", category: "cyber" },
};

describe("Anthropic raw SSE parsing", () => {
	it("forwards parsed provider stream events in order", async () => {
		const model = getModel("anthropic", "claude-haiku-4-5");
		const providerEvents: unknown[] = [];
		const eventModels: Model<Api>[] = [];
		const result = await streamAnthropic(
			model,
			normalizeContext({ messages: [{ role: "user", content: "Hello", timestamp: 1 }] }),
			{
				client: createFakeAnthropicClient(createSseResponse(minimalAnthropicEvents)),
				onProviderStreamEvent: async (event, eventModel) => {
					await Promise.resolve();
					providerEvents.push(event);
					eventModels.push(eventModel);
				},
			},
		).result();

		expect(result.stopReason).toBe("stop");
		expect(providerEvents.map((event) => (event as { type: string }).type)).toEqual([
			"message_start",
			"content_block_start",
			"content_block_delta",
			"content_block_stop",
			"message_delta",
			"message_stop",
		]);
		expect(eventModels).toEqual([model, model, model, model, model, model]);
	});

	it("keeps signed thinking replayable when a proxy relabels the model", async () => {
		// Regression test for earendil-works/pi#9188.
		const model = getModel("anthropic", "claude-opus-5");
		const responseModel = "kimi-for-coding";
		const initialContext = normalizeContext({
			messages: [{ role: "user", content: "Hello", timestamp: 1 }],
		});
		const first = await streamAnthropic(model, initialContext, {
			client: createFakeAnthropicClient(
				createResponseModelSseResponse(responseModel, {
					type: "thinking",
					thinking: "reasoning",
					signature: "signature",
				}),
			),
		}).result();

		expect(first.model).toBe(model.id);
		expect(first.responseModel).toBe(responseModel);

		const transformed = transformMessages([...initialContext.messages, first], model);
		const replayedAssistant = transformed.find((message) => message.role === "assistant");
		expect(replayedAssistant?.content).toEqual([
			{ type: "thinking", thinking: "reasoning", thinkingSignature: "signature" },
		]);
	});

	it("uses a returned fallback model for cost attribution", async () => {
		const fallbackModel = "fallback-model";
		const model: Model<"anthropic-messages"> = {
			...getModel("anthropic", "claude-opus-5"),
			compat: {
				allowedFallbackModels: [
					{
						provider: "anthropic",
						model: fallbackModel,
						cost: { input: 3, output: 5, cacheRead: 0, cacheWrite: 0 },
					},
				],
			},
		};
		const result = await streamAnthropic(
			model,
			normalizeContext({ messages: [{ role: "user", content: "Hello", timestamp: 1 }] }),
			{
				client: createFakeAnthropicClient(
					createResponseModelSseResponse(fallbackModel, { type: "text", text: "done" }),
				),
			},
		).result();

		expect(result.model).toBe(model.id);
		expect(result.responseModel).toBe(fallbackModel);
		expect(result.usage.cost.input).toBeCloseTo(0.0003, 10);
		expect(result.usage.cost.output).toBeCloseTo(0.0001, 10);
	});

	it("continues after a mid-output fallback and replays its boundary", async () => {
		// Regression test for earendil-works/pi#9074.
		const model = withFallbackModels(getModel("anthropic", "claude-opus-5"), [
			{ model: "claude-opus-4-8", input: 3, output: 5 },
		]);
		const stream = streamAnthropic(model, normalizeContext({ messages: [userHello] }), {
			retainsFinalMessage: true,
			client: createFakeAnthropicClient(
				createEventResponse([
					{
						type: "message_start",
						message: {
							id: "msg_fallback",
							model: "claude-opus-5",
							usage: { input_tokens: 100, output_tokens: 0 },
						},
					},
					...blockEvents(0, { type: "thinking", thinking: "", signature: "" }, [
						{ type: "thinking_delta", thinking: "declined reasoning" },
						{ type: "signature_delta", signature: "sig-a" },
					]),
					...blockEvents(1, { type: "redacted_thinking", data: "redacted-a" }),
					...blockEvents(2, { type: "text", text: "" }, [{ type: "text_delta", text: "partial" }]),
					...blockEvents(3, fallbackToOpus48),
					...blockEvents(4, { type: "thinking", thinking: "", signature: "" }, [
						{ type: "thinking_delta", thinking: "fallback reasoning" },
						{ type: "signature_delta", signature: "sig-b" },
					]),
					...blockEvents(5, { type: "tool_use", id: "toolu_b", name: "edit", input: {} }, [
						{ type: "input_json_delta", partial_json: '{"path":"a"}' },
					]),
					{
						type: "message_delta",
						delta: { stop_reason: "tool_use" },
						usage: { input_tokens: 100, output_tokens: 20 },
					},
					{ type: "message_stop" },
				]),
			),
		});
		const events: AssistantMessageEvent[] = [];
		for await (const event of stream) events.push(event);
		const first = await stream.result();

		expect(first.stopReason).toBe("toolUse");
		expect(first.model).toBe("claude-opus-5");
		expect(first.responseModel).toBe("claude-opus-4-8");
		expect(first.usage.cost.input).toBeCloseTo(0.0003, 10);
		expect(first.usage.cost.output).toBeCloseTo(0.0001, 10);
		expect(first.content).toEqual([
			{ type: "thinking", thinking: "declined reasoning", thinkingSignature: "sig-a" },
			{ type: "thinking", thinking: "[Reasoning redacted]", thinkingSignature: "redacted-a", redacted: true },
			{ type: "text", text: "partial" },
			{ type: "thinking", thinking: "fallback reasoning", thinkingSignature: "sig-b" },
			{ type: "toolCall", id: "toolu_b", name: "edit", arguments: { path: "a" } },
		]);
		expect(first.anthropicFallbacks).toEqual([{ position: 3, block: fallbackToOpus48 }]);
		// Streamed content indices still address the final content array.
		expect(
			events.flatMap((event) =>
				event.type === "thinking_end" || event.type === "text_end" || event.type === "toolcall_end"
					? [event.contentIndex]
					: [],
			),
		).toEqual([0, 1, 2, 3, 4]);

		// The replay request has no `fallbacks`; historical markers are still echoed.
		const snapshot = structuredClone(first);
		const payload = await captureReplayPayload(withFallbackModels(getModel("anthropic", "claude-opus-5"), []), [
			userHello,
			first,
			{
				role: "toolResult",
				toolCallId: "toolu_b",
				toolName: "edit",
				content: [{ type: "text", text: "ok" }],
				isError: false,
				timestamp: 2,
			},
		]);
		expect(first).toEqual(snapshot);
		expect(payload.fallbacks).toBeUndefined();
		expect(assistantPayloadContent(payload)).toEqual([
			[
				{ type: "text", text: "partial" },
				fallbackToOpus48,
				{ type: "thinking", thinking: "fallback reasoning", signature: "sig-b" },
				{ type: "tool_use", id: "toolu_b", name: "edit", input: { path: "a" } },
			],
		]);
		const toolResults = payload.messages.flatMap((message) =>
			Array.isArray(message.content) ? message.content.filter((block) => block.type === "tool_result") : [],
		);
		expect(toolResults).toEqual([expect.objectContaining({ tool_use_id: "toolu_b", is_error: false })]);
	});

	it("tracks every fallback hop and prices the last serving model", async () => {
		// Regression test for earendil-works/pi#9074.
		const model = withFallbackModels(getModel("anthropic", "claude-opus-5"), [
			{ model: "model-b", input: 3, output: 5 },
			{ model: "model-c", input: 7, output: 11 },
		]);
		const hopToB = {
			type: "fallback",
			from: { model: "claude-opus-5" },
			to: { model: "model-b" },
			trigger: { type: "refusal", category: "bio" },
		};
		const hopToC = { type: "fallback", from: { model: "model-b" }, to: { model: "model-c" }, trigger: null };
		const first = await streamAnthropic(model, normalizeContext({ messages: [userHello] }), {
			retainsFinalMessage: true,
			client: createFakeAnthropicClient(
				createEventResponse([
					{
						type: "message_start",
						message: { id: "msg_hops", model: "claude-opus-5", usage: { input_tokens: 100, output_tokens: 0 } },
					},
					...blockEvents(0, hopToB),
					...blockEvents(1, { type: "thinking", thinking: "b reasoning", signature: "sig-b" }),
					...blockEvents(2, { type: "text", text: "B text" }),
					...blockEvents(3, hopToC),
					...blockEvents(4, { type: "redacted_thinking", data: "redacted-c" }),
					...blockEvents(5, { type: "thinking", thinking: "c reasoning", signature: "sig-c" }),
					...blockEvents(6, { type: "text", text: "C text" }),
					{
						type: "message_delta",
						delta: { stop_reason: "end_turn" },
						usage: { input_tokens: 100, output_tokens: 20 },
					},
					{ type: "message_stop" },
				]),
			),
		}).result();

		expect(first.stopReason).toBe("stop");
		expect(first.model).toBe("claude-opus-5");
		expect(first.responseModel).toBe("model-c");
		expect(first.usage.cost.input).toBeCloseTo(0.0007, 10);
		expect(first.usage.cost.output).toBeCloseTo(0.00022, 10);
		expect(first.anthropicFallbacks).toEqual([
			{ position: 0, block: hopToB },
			{ position: 2, block: hopToC },
		]);

		const payload = await captureReplayPayload(model, [userHello, first]);
		expect(payload.fallbacks).toEqual([{ model: "model-b" }, { model: "model-c" }]);
		expect(assistantPayloadContent(payload)).toEqual([
			[
				hopToB,
				{ type: "text", text: "B text" },
				hopToC,
				{ type: "redacted_thinking", data: "redacted-c" },
				{ type: "thinking", thinking: "c reasoning", signature: "sig-c" },
				{ type: "text", text: "C text" },
			],
		]);
	});

	it("rebases fallback markers when replay drops or skips blocks", async () => {
		// Regression test for earendil-works/pi#9074.
		const hopToB = { type: "fallback" as const, from: { model: "claude-opus-5" }, to: { model: "model-b" } };
		const hopToC = { type: "fallback" as const, from: { model: "model-b" }, to: { model: "model-c" } };
		const assistant = (
			content: AssistantMessage["content"],
			anthropicFallbacks: AssistantMessage["anthropicFallbacks"],
		): AssistantMessage => ({
			role: "assistant",
			content,
			anthropicFallbacks,
			api: "anthropic-messages",
			provider: "anthropic",
			model: "claude-opus-5",
			responseModel: "model-c",
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
			timestamp: 1,
		});
		// Adjacent markers precede an empty text block that the converter skips.
		const adjacent = assistant(
			[
				{ type: "thinking", thinking: "", thinkingSignature: "sig-a" },
				{ type: "thinking", thinking: "abandoned", thinkingSignature: "" },
				{ type: "text", text: "" },
				{ type: "text", text: "A" },
				{ type: "thinking", thinking: "", thinkingSignature: "sig-c" },
				{ type: "thinking", thinking: "c reasoning", thinkingSignature: "sig-c2" },
				{ type: "text", text: "C" },
			],
			[
				{ position: 2, block: hopToB },
				{ position: 2, block: hopToC },
			],
		);
		// A trailing marker with no content after it.
		const trailing = assistant(
			[
				{ type: "thinking", thinking: "declined", thinkingSignature: "sig-a" },
				{ type: "text", text: "A" },
			],
			[{ position: 2, block: hopToB }],
		);
		const messages: Message[] = [
			userHello,
			adjacent,
			{ role: "user", content: "Next", timestamp: 2 },
			trailing,
			{ role: "user", content: "Again", timestamp: 3 },
		];
		const snapshot = structuredClone(messages);

		const sameModel = await captureReplayPayload(getModel("anthropic", "claude-opus-5"), messages);
		expect(assistantPayloadContent(sameModel)).toEqual([
			[
				hopToB,
				hopToC,
				{ type: "text", text: "A" },
				{ type: "thinking", thinking: "", signature: "sig-c" },
				{ type: "thinking", thinking: "c reasoning", signature: "sig-c2" },
				{ type: "text", text: "C" },
			],
			[{ type: "text", text: "A" }, hopToB],
		]);

		// Cross-model replay drops empty signed thinking and turns the rest into text.
		const crossModel = await captureReplayPayload(getModel("anthropic", "claude-haiku-4-5"), messages);
		expect(assistantPayloadContent(crossModel)).toEqual([
			[
				hopToB,
				hopToC,
				{ type: "text", text: "A" },
				{ type: "text", text: "c reasoning" },
				{ type: "text", text: "C" },
			],
			[{ type: "text", text: "A" }, hopToB],
		]);

		// Other APIs receive neither markers nor abandoned thinking as text.
		const openai = transformMessages(messages, getModel("openai", "gpt-4o"));
		const openaiAssistants = openai.filter((message) => message.role === "assistant");
		expect(openaiAssistants.map((message) => message.anthropicFallbacks)).toEqual([undefined, undefined]);
		expect(openaiAssistants.map((message) => message.content)).toEqual([
			[
				{ type: "text", text: "" },
				{ type: "text", text: "A" },
				{ type: "text", text: "c reasoning" },
				{ type: "text", text: "C" },
			],
			[{ type: "text", text: "A" }],
		]);
		expect(messages).toEqual(snapshot);
	});

	it("fails a mid-output fallback unless the caller retains the final message", async () => {
		// Regression test for earendil-works/pi#9074: callers that rebuild messages from reduced
		// events would lose the marker, so they keep the released fail-closed behavior.
		const model = withFallbackModels(getModel("anthropic", "claude-opus-5"), [
			{ model: "claude-opus-4-8", input: 3, output: 5 },
		]);
		const prefixes = [
			blockEvents(0, { type: "text", text: "" }, [{ type: "text_delta", text: "partial" }]),
			blockEvents(0, { type: "thinking", thinking: "", signature: "" }, [
				{ type: "thinking_delta", thinking: "declined reasoning" },
				{ type: "signature_delta", signature: "sig-a" },
			]),
		];
		for (const prefix of prefixes) {
			const result = await streamAnthropic(model, normalizeContext({ messages: [userHello] }), {
				client: createFakeAnthropicClient(
					createEventResponse([
						{
							type: "message_start",
							message: {
								id: "msg_prefix",
								model: "claude-opus-5",
								usage: { input_tokens: 100, output_tokens: 0 },
							},
						},
						...prefix,
						...blockEvents(1, fallbackToOpus48),
						...blockEvents(2, { type: "text", text: "fallback answer" }),
						{ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 5 } },
						{ type: "message_stop" },
					]),
				),
			}).result();

			expect(result.stopReason).toBe("error");
			expect(result.errorMessage).toBe("Anthropic performed an unsupported mid-output model fallback");
			expect(result.anthropicFallbacks).toBeUndefined();
			expect(result.responseModel).toBeUndefined();
			expect(result.content).toHaveLength(1);
		}
	});

	it("forwards retainsFinalMessage through streamSimple", async () => {
		// Regression test for earendil-works/pi#9074: buildBaseOptions must not drop the opt-in.
		const model = withFallbackModels(getModel("anthropic", "claude-opus-5"), [
			{ model: "claude-opus-4-8", input: 3, output: 5 },
		]);
		const fetchImpl: typeof fetch = async () =>
			createEventResponse([
				{
					type: "message_start",
					message: { id: "msg_simple", model: "claude-opus-5", usage: { input_tokens: 100, output_tokens: 0 } },
				},
				...blockEvents(0, { type: "text", text: "partial" }),
				...blockEvents(1, fallbackToOpus48),
				...blockEvents(2, { type: "text", text: "fallback answer" }),
				{ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 5 } },
				{ type: "message_stop" },
			]);
		const run = (retainsFinalMessage?: boolean) =>
			streamAnthropicSimple(model, normalizeContext({ messages: [userHello] }), {
				apiKey: "test-key",
				cacheRetention: "none",
				fetch: fetchImpl,
				retainsFinalMessage,
			}).result();

		const retained = await run(true);
		expect(retained.stopReason).toBe("stop");
		expect(retained.responseModel).toBe("claude-opus-4-8");
		expect(retained.anthropicFallbacks).toEqual([{ position: 1, block: fallbackToOpus48 }]);
		expect(retained.content).toEqual([
			{ type: "text", text: "partial" },
			{ type: "text", text: "fallback answer" },
		]);

		const rebuilt = await run();
		expect(rebuilt.stopReason).toBe("error");
		expect(rebuilt.errorMessage).toBe("Anthropic performed an unsupported mid-output model fallback");
	});

	it("fails closed when Anthropic falls back after a tool call", async () => {
		// Regression test for earendil-works/pi#9074: the declined model's tool call must not run,
		// even when the caller retains the final message.
		const run = (retainsFinalMessage?: boolean) =>
			streamAnthropic(getModel("anthropic", "claude-opus-5"), normalizeContext({ messages: [userHello] }), {
				retainsFinalMessage,
				client: createFakeAnthropicClient(
					createEventResponse([
						{
							type: "message_start",
							message: { id: "msg_tool", model: "claude-opus-5", usage: { input_tokens: 1, output_tokens: 0 } },
						},
						...blockEvents(0, { type: "text", text: "partial" }),
						...blockEvents(1, { type: "tool_use", id: "toolu_a", name: "edit", input: {} }),
						...blockEvents(2, fallbackToOpus48),
						...blockEvents(3, { type: "text", text: "fallback answer" }),
						{ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 5 } },
						{ type: "message_stop" },
					]),
				),
			}).result();

		const unflagged = await run();
		expect(unflagged.stopReason).toBe("error");
		expect(unflagged.errorMessage).toBe("Anthropic performed an unsupported mid-output model fallback");

		const retained = await run(true);
		expect(retained.stopReason).toBe("error");
		expect(retained.errorMessage).toBe("Anthropic performed an unsupported model fallback after a tool call");
		expect(retained.anthropicFallbacks).toBeUndefined();
	});

	it("handles an early fallback and replays its markers with the fallback beta", async () => {
		// Regression test for earendil-works/pi#9074: the beta is needed to replay markers.
		const model = withFallbackModels(getModel("anthropic", "claude-opus-5"), [
			{ model: "claude-opus-4-8", input: 3, output: 5 },
		]);
		const first = await streamAnthropic(model, normalizeContext({ messages: [userHello] }), {
			client: createFakeAnthropicClient(
				createEventResponse([
					{
						type: "message_start",
						message: {
							id: "msg_early",
							model: "claude-opus-4-8",
							usage: { input_tokens: 100, output_tokens: 0 },
						},
					},
					...blockEvents(0, fallbackToOpus48),
					...blockEvents(1, { type: "thinking", thinking: "b reasoning", signature: "sig-b" }),
					...blockEvents(2, { type: "text", text: "B text" }),
					{
						type: "message_delta",
						delta: { stop_reason: "end_turn" },
						usage: { input_tokens: 100, output_tokens: 20 },
					},
					{ type: "message_stop" },
				]),
			),
		}).result();

		expect(first.model).toBe("claude-opus-5");
		expect(first.responseModel).toBe("claude-opus-4-8");
		expect(first.usage.cost.input).toBeCloseTo(0.0003, 10);
		expect(first.usage.cost.output).toBeCloseTo(0.0001, 10);
		expect(first.anthropicFallbacks).toEqual([{ position: 0, block: fallbackToOpus48 }]);

		// The replay model requests no fallbacks, but the replayed markers still need the beta.
		const replayModel = withFallbackModels(getModel("anthropic", "claude-opus-5"), []);
		const payload = await captureReplayPayload(replayModel, [userHello, first]);
		expect(payload.fallbacks).toBeUndefined();
		expect(payload.betas).toContain("server-side-fallback-2026-07-01");
		expect(assistantPayloadContent(payload)).toEqual([
			[
				fallbackToOpus48,
				{ type: "thinking", thinking: "b reasoning", signature: "sig-b" },
				{ type: "text", text: "B text" },
			],
		]);

		// No marker is replayed for ordinary, errored, or other-provider history.
		for (const messages of [
			[userHello],
			[userHello, { ...first, stopReason: "error" as const }],
			[userHello, { ...first, provider: "other-provider" }],
		]) {
			const betas = (await captureReplayPayload(replayModel, messages)).betas ?? [];
			expect(betas).not.toContain("server-side-fallback-2026-07-01");
		}
		// An explicit beta header still replaces or suppresses the defaults.
		const suppressed = await captureReplayPayload(replayModel, [userHello, first], {
			headers: { "anthropic-beta": null },
		});
		expect(suppressed.betas).toBeUndefined();
	});

	it("aborts during a fallback continuation", async () => {
		// Regression test for earendil-works/pi#9074.
		const controller = new AbortController();
		const result = await streamAnthropic(
			getModel("anthropic", "claude-opus-5"),
			normalizeContext({ messages: [userHello] }),
			{
				signal: controller.signal,
				retainsFinalMessage: true,
				onProviderStreamEvent: (event) => {
					const data = event as { type: string; content_block?: { type: string } };
					if (data.type === "content_block_start" && data.content_block?.type === "fallback") controller.abort();
				},
				client: createFakeAnthropicClient(
					createEventResponse([
						{
							type: "message_start",
							message: { id: "msg_abort", model: "claude-opus-5", usage: { input_tokens: 1, output_tokens: 0 } },
						},
						...blockEvents(0, { type: "text", text: "partial" }),
						...blockEvents(1, fallbackToOpus48),
						...blockEvents(2, { type: "text", text: "fallback answer" }),
						{ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 5 } },
						{ type: "message_stop" },
					]),
				),
			},
		).result();

		expect(result.stopReason).toBe("aborted");
	});

	it("fails when the stream ends after a fallback marker", async () => {
		// Regression test for earendil-works/pi#9074.
		const model = getModel("anthropic", "claude-opus-5");
		const response = createSseResponse([
			{
				event: "message_start",
				data: JSON.stringify({
					type: "message_start",
					message: {
						id: "msg_fallback",
						model: "claude-opus-5",
						usage: { input_tokens: 1, output_tokens: 0 },
					},
				}),
			},
			{
				event: "content_block_start",
				data: JSON.stringify({
					type: "content_block_start",
					index: 0,
					content_block: { type: "text", text: "partial" },
				}),
			},
			{ event: "content_block_stop", data: JSON.stringify({ type: "content_block_stop", index: 0 }) },
			{
				event: "content_block_start",
				data: JSON.stringify({
					type: "content_block_start",
					index: 1,
					content_block: {
						type: "fallback",
						from: { model: "claude-opus-5" },
						to: { model: "claude-opus-4-8" },
					},
				}),
			},
		]);

		const result = await streamAnthropic(
			model,
			normalizeContext({ messages: [{ role: "user", content: "Hello", timestamp: 1 }] }),
			{ retainsFinalMessage: true, client: createFakeAnthropicClient(response) },
		).result();

		expect(result.stopReason).toBe("error");
		expect(result.errorMessage).toBe("Anthropic stream ended before message_stop");
	});

	it("forces streaming after an onPayload replacement", async () => {
		let streaming: boolean | undefined;
		const client = {
			beta: {
				messages: {
					create: (params: { stream?: boolean }) => {
						streaming = params.stream;
						return { asResponse: async () => createSseResponse(minimalAnthropicEvents) };
					},
				},
			},
		} as unknown as Anthropic;

		await streamAnthropic(
			getModel("anthropic", "claude-fable-5-1"),
			normalizeContext({ messages: [{ role: "user", content: "Hello", timestamp: 1 }] }),
			{
				client,
				onPayload: (payload) => ({ ...(payload as Record<string, unknown>), stream: false }),
			},
		).result();

		expect(streaming).toBe(true);
	});

	it("omits the interleaved-thinking beta when thinking is disabled", async () => {
		let betaFeatures: string[] | undefined;
		const client = {
			beta: {
				messages: {
					create: (params: { betas?: string[] }) => {
						betaFeatures = params.betas;
						return { asResponse: async () => createSseResponse(minimalAnthropicEvents) };
					},
				},
			},
		} as unknown as Anthropic;

		await streamAnthropic(
			getModel("openrouter", "anthropic/claude-haiku-4.5"),
			normalizeContext({ messages: [{ role: "user", content: "Hello", timestamp: 1 }] }),
			{ client, thinkingEnabled: false },
		).result();

		expect(betaFeatures ?? []).not.toContain("interleaved-thinking-2025-05-14");
	});

	it("passes managed beta features to injected clients", async () => {
		let betaFeatures: string[] | undefined;
		const client = {
			beta: {
				messages: {
					create: (params: { betas?: string[] }) => {
						betaFeatures = params.betas;
						return { asResponse: async () => createSseResponse(minimalAnthropicEvents) };
					},
				},
			},
		} as unknown as Anthropic;

		const result = await streamAnthropic(
			getModel("anthropic", "claude-fable-5-1"),
			normalizeContext({ messages: [{ role: "user", content: "Hello", timestamp: 1 }] }),
			{ client },
		).result();

		expect(result.stopReason).toBe("stop");
		expect(betaFeatures).toContain("mid-conversation-output-config-2026-07-01");
		expect(betaFeatures).toContain("thinking-binding-controls-2026-08-01");
	});

	it("uses the serving model input transformations from the final stream event", async () => {
		const events = minimalAnthropicEvents.map((event) => ({ ...event }));
		events[0].data = JSON.stringify({
			type: "message_start",
			message: {
				id: "msg_transformations",
				model: "claude-fable-5-1",
				usage: { input_tokens: 12, output_tokens: 0 },
				input_transformations: [
					{ type: "thinking_dropped", path: "messages.1.content.0", reason: "prefix_binding_mismatch" },
				],
			},
		});
		const delta = JSON.parse(events[4].data) as Record<string, unknown>;
		delta.input_transformations = [
			{ type: "thinking_dropped", path: "messages.3.content.0", reason: "model_binding_mismatch" },
		];
		events[4].data = JSON.stringify(delta);

		const result = await streamAnthropic(
			getModel("anthropic", "claude-fable-5-1"),
			normalizeContext({ messages: [{ role: "user", content: "Hello", timestamp: 1 }] }),
			{ client: createFakeAnthropicClient(createSseResponse(events)) },
		).result();

		expect(result.diagnostics).toEqual([
			{
				type: "anthropic_input_transformations",
				timestamp: expect.any(Number),
				details: {
					transformations: [
						{
							type: "thinking_dropped",
							path: "messages.3.content.0",
							reason: "model_binding_mismatch",
						},
					],
				},
			},
		]);
	});
	it("repairs malformed SSE JSON and malformed streamed tool JSON", async () => {
		const model = getModel("anthropic", "claude-haiku-4-5");
		const context = normalizeContext({
			messages: [{ role: "user", content: "Use the edit tool.", timestamp: Date.now() }],
			tools: [
				{
					name: "edit",
					description: "Edit a file.",
					parameters: Type.Object({
						path: Type.String(),
						text: Type.String(),
					}),
				},
			],
		});

		const malformedToolJsonDelta = String.raw`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"A\H\",\"text\":\"col1	col2\"}"}}`;

		const response = createSseResponse([
			{
				event: "message_start",
				data: JSON.stringify({
					type: "message_start",
					message: {
						id: "msg_test",
						usage: {
							input_tokens: 12,
							output_tokens: 0,
							cache_read_input_tokens: 0,
							cache_creation_input_tokens: 0,
						},
					},
				}),
			},
			{
				event: "content_block_start",
				data: JSON.stringify({
					type: "content_block_start",
					index: 0,
					content_block: {
						type: "tool_use",
						id: "toolu_test",
						name: "edit",
						input: {},
					},
				}),
			},
			{ event: "content_block_delta", data: malformedToolJsonDelta },
			{
				event: "content_block_stop",
				data: JSON.stringify({ type: "content_block_stop", index: 0 }),
			},
			{
				event: "message_delta",
				data: JSON.stringify({
					type: "message_delta",
					delta: { stop_reason: "tool_use" },
					usage: {
						input_tokens: 12,
						output_tokens: 5,
						cache_read_input_tokens: 0,
						cache_creation_input_tokens: 0,
					},
				}),
			},
			{
				event: "message_stop",
				data: JSON.stringify({ type: "message_stop" }),
			},
		]);

		const stream = streamAnthropic(model, context, {
			client: createFakeAnthropicClient(response),
		});
		const result = await stream.result();

		expect(result.stopReason).toBe("toolUse");
		expect(result.errorMessage).toBeUndefined();

		const toolCall = result.content.find((block): block is ToolCall => block.type === "toolCall");
		expect(toolCall).toBeDefined();
		expect(toolCall?.arguments).toEqual({
			path: "A\\H",
			text: "col1\tcol2",
		});
	});

	it("preserves content from content_block_start events", async () => {
		const model = getModel("anthropic", "claude-haiku-4-5");
		const context = normalizeContext({
			messages: [{ role: "user", content: "Say hello.", timestamp: Date.now() }],
		});
		const response = createSseResponse([
			{
				event: "message_start",
				data: JSON.stringify({
					type: "message_start",
					message: {
						id: "msg_initial_content",
						usage: {
							input_tokens: 12,
							output_tokens: 0,
							cache_read_input_tokens: 0,
							cache_creation_input_tokens: 0,
						},
					},
				}),
			},
			{
				event: "content_block_start",
				data: JSON.stringify({
					type: "content_block_start",
					index: 0,
					content_block: { type: "text", text: "Initial text" },
				}),
			},
			{
				event: "content_block_delta",
				data: JSON.stringify({
					type: "content_block_delta",
					index: 0,
					delta: { type: "text_delta", text: " plus delta" },
				}),
			},
			{ event: "content_block_stop", data: JSON.stringify({ type: "content_block_stop", index: 0 }) },
			{
				event: "content_block_start",
				data: JSON.stringify({
					type: "content_block_start",
					index: 1,
					content_block: {
						type: "thinking",
						thinking: "Initial thinking",
						signature: "initial signature",
					},
				}),
			},
			{
				event: "content_block_delta",
				data: JSON.stringify({
					type: "content_block_delta",
					index: 1,
					delta: { type: "thinking_delta", thinking: " plus delta" },
				}),
			},
			{
				event: "content_block_delta",
				data: JSON.stringify({
					type: "content_block_delta",
					index: 1,
					delta: { type: "signature_delta", signature: " plus delta" },
				}),
			},
			{ event: "content_block_stop", data: JSON.stringify({ type: "content_block_stop", index: 1 }) },
			{
				event: "message_delta",
				data: JSON.stringify({
					type: "message_delta",
					delta: { stop_reason: "end_turn" },
					usage: {
						input_tokens: 12,
						output_tokens: 5,
						cache_read_input_tokens: 0,
						cache_creation_input_tokens: 0,
					},
				}),
			},
			{ event: "message_stop", data: JSON.stringify({ type: "message_stop" }) },
		]);

		const stream = streamAnthropic(model, context, {
			client: createFakeAnthropicClient(response),
		});
		const result = await stream.result();

		expect(result.content).toEqual([
			{ type: "text", text: "Initial text plus delta" },
			{
				type: "thinking",
				thinking: "Initial thinking plus delta",
				thinkingSignature: "initial signature plus delta",
			},
		]);
	});

	it("preserves refusal stop details from message_delta", async () => {
		const model = getModel("anthropic", "claude-fable-5");
		const context = normalizeContext({
			messages: [{ role: "user", content: "blocked request", timestamp: Date.now() }],
		});
		const explanation =
			"This request triggered restrictions on violative cyber content and was blocked under Anthropic's Usage Policy. To learn more, provide feedback, or request an exemption based on how you use Claude, visit our help center: https://support.claude.com/en/articles/14604842-real-time-cyber-safeguards-on-claude.";
		const response = createSseResponse([
			{
				event: "message_start",
				data: JSON.stringify({
					type: "message_start",
					message: {
						id: "msg_01XFUDYJgAACzvnptvVoYEL",
						usage: {
							input_tokens: 412,
							output_tokens: 0,
							cache_read_input_tokens: 0,
							cache_creation_input_tokens: 0,
						},
					},
				}),
			},
			{
				event: "message_delta",
				data: JSON.stringify({
					type: "message_delta",
					delta: {
						stop_reason: "refusal",
						stop_details: {
							type: "refusal",
							category: "cyber",
							explanation,
						},
					},
					usage: {
						input_tokens: 412,
						output_tokens: 0,
						cache_read_input_tokens: 0,
						cache_creation_input_tokens: 0,
					},
				}),
			},
			{
				event: "message_stop",
				data: JSON.stringify({ type: "message_stop" }),
			},
		]);

		const stream = streamAnthropic(model, context, {
			client: createFakeAnthropicClient(response),
		});
		const result = await stream.result();

		expect(result.stopReason).toBe("error");
		expect(result.rawStopReason).toBe("refusal");
		expect(result.errorMessage).toBe(explanation);
	});

	it("preserves sensitive stop reasons with a descriptive error message", async () => {
		const model = getModel("anthropic", "claude-haiku-4-5");
		const context = normalizeContext({
			messages: [{ role: "user", content: "blocked request", timestamp: Date.now() }],
		});
		const response = createSseResponse([
			{
				event: "message_start",
				data: JSON.stringify({
					type: "message_start",
					message: {
						id: "msg_sensitive",
						usage: {
							input_tokens: 12,
							output_tokens: 0,
							cache_read_input_tokens: 0,
							cache_creation_input_tokens: 0,
						},
					},
				}),
			},
			{
				event: "message_delta",
				data: JSON.stringify({
					type: "message_delta",
					delta: { stop_reason: "sensitive" },
					usage: {
						input_tokens: 12,
						output_tokens: 0,
						cache_read_input_tokens: 0,
						cache_creation_input_tokens: 0,
					},
				}),
			},
			{
				event: "message_stop",
				data: JSON.stringify({ type: "message_stop" }),
			},
		]);

		const stream = streamAnthropic(model, context, {
			client: createFakeAnthropicClient(response),
		});
		const result = await stream.result();

		expect(result.stopReason).toBe("error");
		expect(result.rawStopReason).toBe("sensitive");
		expect(result.errorMessage).toBe("Provider stopped with: sensitive");
	});

	it("treats message_delta without usage as a no-op for usage accumulation", async () => {
		const model = getModel("anthropic", "claude-haiku-4-5");
		const context = normalizeContext({
			messages: [{ role: "user", content: "Say hello.", timestamp: Date.now() }],
		});
		const response = createSseResponse(
			minimalAnthropicEvents.map((event) =>
				event.event === "message_delta"
					? {
							event: "message_delta",
							data: JSON.stringify({ type: "message_delta", delta: { stop_reason: "end_turn" } }),
						}
					: event,
			),
		);

		const stream = streamAnthropic(model, context, {
			client: createFakeAnthropicClient(response),
		});
		const result = await stream.result();

		expect(result.stopReason).toBe("stop");
		expect(result.errorMessage).toBeUndefined();
		expect(result.content).toEqual([{ type: "text", text: "Hello" }]);
		expect(result.usage.input).toBe(12);
		expect(result.usage.totalTokens).toBe(12);
	});

	it("ignores unknown SSE events after message_stop", async () => {
		const model = getModel("anthropic", "claude-haiku-4-5");
		const context = normalizeContext({
			messages: [{ role: "user", content: "Say hello.", timestamp: Date.now() }],
		});
		const response = createSseResponse([
			...minimalAnthropicEvents,
			{ event: "done", data: "[DONE]" },
			{ event: "proxy.stats", data: "not json" },
		]);

		const stream = streamAnthropic(model, context, {
			client: createFakeAnthropicClient(response),
		});
		const result = await stream.result();

		expect(result.stopReason).toBe("stop");
		expect(result.errorMessage).toBeUndefined();
		expect(result.content).toEqual([{ type: "text", text: "Hello" }]);
	});
});

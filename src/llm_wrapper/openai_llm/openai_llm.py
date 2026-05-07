from typing import Any

from openai import OpenAI, AsyncOpenAI
from openai.types import ChatModel
from openai.types.chat import ChatCompletionMessageParam
from pydantic import Field

from llm_wrapper.llm import LLM, to_chat_messages, LLMInput, ChatResult, ToolChoice
from llm_wrapper.openai_llm.tool import Tool
from llm_wrapper.openai_llm.utils import async_chat_result_from_openai, to_openai_message, chat_result_from_openai, \
    tools_to_openai, \
    tool_choice_to_openai, tool_to_openai
from llm_wrapper.utils import filter_kwargs_by_method


class OpenAILLM(LLM):
    model: str | ChatModel = "gpt-5"
    api_key: str | None = None
    base_url: str | None = None
    max_retries: int = 2
    timeout: float = 20  # seconds
    stream_include_usage: bool = Field(
        default=False,
        description="If set, the token usage will return at the end of stream."
                    "Refer to ChatCompletionChunk.usage and ChatCompletionStreamOptionsParam for more info"
    )

    def chat(self, messages: LLMInput, **kwargs: Any) -> ChatResult:
        # Not specify stream=False.
        # Maybe, we want to stream in the background and merge the streamed chunks at the end.
        return self._chat_(messages, **kwargs).merge_chunk()

    def stream_chat(self, messages: LLMInput, **kwargs: Any) -> ChatResult:
        return self._chat_(messages, **{**kwargs, "stream": True})

    async def async_chat(self, messages: LLMInput, **kwargs: Any) -> ChatResult:
        return (await self._async_chat_(messages, **kwargs)).merge_chunk()

    async def async_stream_chat(self, messages: LLMInput, **kwargs: Any) -> ChatResult:
        return await self._async_chat_(messages, **{**kwargs, "stream": True})

    def _prepare_chat_kwargs(self, messages: LLMInput, **kwargs: Any) -> list[ChatCompletionMessageParam]:
        messages = to_chat_messages(messages)
        messages = self.add_system_message(messages, kwargs.get('system_prompt'))
        return list(map(to_openai_message, messages))

    def _chat_(self, messages: LLMInput, **kwargs: Any) -> ChatResult:
        openai_messages = self._prepare_chat_kwargs(messages, **kwargs)
        kwargs = self.get_chat_kwargs(**kwargs)
        assert not (kwargs.get("stream") and kwargs["n"] > 1)
        resp = self.client.chat.completions.create(messages=openai_messages, **kwargs)
        return chat_result_from_openai(resp)

    async def _async_chat_(self, messages: LLMInput, **kwargs: Any) -> ChatResult:
        openai_messages = self._prepare_chat_kwargs(messages, **kwargs)
        kwargs = self.get_chat_kwargs(**kwargs)
        assert not (kwargs.get("stream") and kwargs["n"] > 1)
        resp = await self.async_client.chat.completions.create(messages=openai_messages, **kwargs)
        return async_chat_result_from_openai(resp)

    def get_chat_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        kwargs = {**self.model_dump(exclude_none=True, by_alias=True), **kwargs}

        kwargs["max_tokens"] = kwargs.pop("max_new_tokens")
        if "stream" not in kwargs:
            kwargs["stream"] = kwargs.pop("streaming")

        # Openai doesn't support repetition_penalty,
        # but we can extend it to other models that support it
        # with openai_llm interface compatible server by using 'extra_body'.
        if repetition_penalty := kwargs.pop("repetition_penalty", None):
            kwargs["presence_penalty"] = repetition_penalty
            extra_body = kwargs.get('extra_body', {})
            extra_body.update(repetition_penalty=repetition_penalty)
            kwargs["extra_body"] = extra_body

        if kwargs.get("stream_include_usage"):
            kwargs["stream_options"] = dict(include_usage=True)

        if tools := kwargs.pop("tools", []):
            tools = list(map(Tool.model_validate, tools))
            kwargs["tools"] = tools_to_openai(tools)

        tool_choice: ToolChoice | None = kwargs.pop("tool_choice", None)
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice_to_openai(tool_choice, tools)

        if kwargs.pop("json_mode", False):
            kwargs["response_format"] = {"type": "json_object"}

        return filter_kwargs_by_method(OpenAI(api_key='none').chat.completions.create, kwargs, exclude_none=True)

    @property
    def client(self):
        return OpenAI(api_key=self.api_key,
                      base_url=self.base_url,
                      max_retries=self.max_retries,
                      timeout=self.timeout)

    @property
    def async_client(self):
        return AsyncOpenAI(api_key=self.api_key,
                           base_url=self.base_url,
                           max_retries=self.max_retries,
                           timeout=self.timeout)

    @property
    def openai_tools(self) -> list[dict[str, Any]] | None:
        return list(map(tool_to_openai, self.tools)) if self.tools else None

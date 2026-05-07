from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Union, Sequence, Tuple, List, Iterator, Literal, Any, Callable, AsyncIterator, Optional

from pydantic import Field, BaseModel, field_validator

from llm_wrapper.messages.chat_message import ChatMessage, ChatMessageChunk, Role
from llm_wrapper.messages.utils import to_chat_message, MessageLike
from llm_wrapper.openai_llm.tool import to_tool, Tool
from llm_wrapper.common import TokenUsage
from llm_wrapper.utils import add

LLMInput = Union[MessageLike, Sequence[MessageLike]]
ToolChoiceLiteral = Literal["none", "auto", "required", "any"]
# todo 还有一个类型，在工具列表中取子集，对提高kv缓存利用率有好处，因为可以避免修改 system prompt 的工具列表。
ToolChoice = str | ToolChoiceLiteral | bool


class LLM(BaseModel, ABC):
    model: str

    max_new_tokens: int = Field(
        default=512,
        description="Number of tokens the model can output when generating a response.",
    )

    temperature: float = Field(
        default=0.1,
        description="The temperature to use during generation.",
        ge=0.0,
    )

    streaming: bool = Field(default=False, alias="stream", description="Streaming output.")

    repetition_penalty: float | None = 1

    stop: str | list[str] | None = None

    n: int = Field(default=1, description="How many chat completion choices to generate for each input message.")

    top_p: float | None = None

    system_prompt: str | None = None

    tools: list[Tool] | None = None
    tool_choice: ToolChoice | None = None

    parallel_tool_calls: bool | None = None

    extra_body: Optional[dict] = None

    @field_validator('tools')
    @classmethod
    def validate_tools(cls, tools: List[Tool | Callable]) -> List[Tool]:
        return [to_tool(tool) for tool in tools]

    @abstractmethod
    def chat(self, messages: LLMInput, **kwargs: Any) -> ChatResult:
        ...

    @abstractmethod
    def stream_chat(self, messages: LLMInput, **kwargs: Any) -> ChatResult:
        ...

    @abstractmethod
    async def async_chat(self, messages: LLMInput, **kwargs: Any) -> ChatResult:
        ...

    @abstractmethod
    async def async_stream_chat(self, messages: LLMInput, **kwargs: Any) -> ChatResult:
        ...

    def add_system_message(self, messages: List[ChatMessage], system_prompt: str | None = None
                           ) -> List[ChatMessage]:
        system_prompt = system_prompt or self.system_prompt
        if system_prompt is None:
            return messages

        assert messages[0].role != Role.SYSTEM
        return [ChatMessage(role=Role.SYSTEM, content=system_prompt)] + list(messages)


def to_chat_messages(inp: LLMInput) -> List[ChatMessage]:
    try:
        return [to_chat_message(inp)]  # type: ignore[arg-type]
    except ValueError as e:
        return list(map(to_chat_message, inp))


class ChatResult(BaseModel):
    messages: List[ChatMessage] = Field(default_factory=list)
    """
    When used as LLM's output, len(messages) is LLM.n,
    and when used as Agent's output, messages messages[-1] is the final answer,
    messages[:-1] is intermediate steps, if it has.
    """

    usage: TokenUsage | None = None

    message_stream: Iterator[Tuple[ChatMessageChunk, TokenUsage | None]] | None = Field(
        default=None, description="only return stream of index 0"
    )

    async_message_stream: AsyncIterator[Tuple[ChatMessageChunk, TokenUsage | None]] | None = Field(
        default=None, description="only return stream of index 0"
    )

    message_stream_for_agent: Iterator[ChatMessageChunk | ChatMessage] | None = None
    """
    ChatMessageChunk are final answer or thoughts, ChatMessage are tool calls or observations
    ChatMessageChunk always go before ChatMessage
    """

    def merge_chunk(self) -> ChatResult:
        if self.message_stream:
            assert len(self.messages) == 0
            message_cache: ChatMessageChunk | None = None
            for message_chunk, usage_chunk in self.message_stream:
                message_cache = add(message_cache, message_chunk)
                self.usage = add(self.usage, usage_chunk)
            if message_cache:
                if message_cache.role is None:  # qwen3 官方api的bug
                    message_cache.role = Role.ASSISTANT
                self.messages.append(message_cache.to_message())
            self.message_stream = None

        if self.message_stream_for_agent:
            assert len(self.messages) == 0
            message_cache = None
            for message_or_chunk in self.message_stream_for_agent:
                if isinstance(message_or_chunk, ChatMessageChunk):
                    message_cache = add(message_cache, message_or_chunk)
                else:
                    if message_cache:
                        self.messages.append(message_cache.to_message())
                        message_cache = None
                    self.messages.append(message_or_chunk)
            if message_cache:
                self.messages.append(message_cache.to_message())
            self.message_stream_for_agent = None

        return self

    class Config:
        arbitrary_types_allowed = True

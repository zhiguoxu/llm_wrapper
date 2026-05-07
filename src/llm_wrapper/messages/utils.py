from typing import Union, Tuple, Dict, Any, Iterable

from llm_wrapper.messages.chat_message import TextContent, ImageContent, ImageURL, ChatMessage, Role

MessageLike = Union[ChatMessage, list[str], Tuple[str, str], str, Dict[str, Any]]


def format_content(content: str | Iterable | None) -> str | list[TextContent | ImageContent] | None:
    if content is None or isinstance(content, str):
        return content

    assert isinstance(content, list)

    content_list = []
    for item in content:
        if isinstance(item, str):
            new_content = TextContent(text=item)
        else:
            assert isinstance(item, dict)
            if url := item.get('url'):
                new_content = ImageContent(image_url=ImageURL(url=url))
            elif image_url := item.get('image_url'):
                new_content = ImageContent(image_url=image_url)
            else:
                new_content = TextContent(text=item['text'])
        content_list.append(new_content)

    return content_list


def to_chat_message(message: MessageLike) -> ChatMessage:
    if isinstance(message, ChatMessage):
        return message

    if isinstance(message, str):
        return ChatMessage(role=Role.USER, content=message)

    if isinstance(message, (list, tuple)):
        if len(message) != 2:
            raise ValueError(f"MessageLike type error {message}")

        return ChatMessage(role=Role.from_name(message[0]), content=message[1])

    if isinstance(message, dict):
        content = format_content(message["content"])
        return ChatMessage(role=Role.from_name(message.get("role", Role.USER)), content=content)

    raise ValueError(f"Error chat message: {message}")

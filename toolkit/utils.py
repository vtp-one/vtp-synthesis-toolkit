###
#
from typing import Any
from collections.abc import AsyncIterable, Iterable

from pydantic import BaseModel
from starlette.background import BackgroundTask
from starlette.concurrency import iterate_in_threadpool
from starlette.responses import JSONResponse, StreamingResponse


from langchain_core.load.serializable import Serializable
import json
import rich
from json import JSONEncoder

class LangChainEncoder(JSONEncoder):
    def default(self, o):
        match o:
            case Serializable():
                return o.dict()

            case _:
                pass

        return super().default(o)

class JSONStreamingResponse(StreamingResponse, JSONResponse):
    """StreamingResponse that also render with JSON."""

    def __init__(
        self,
        content: Iterable | AsyncIterable,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        media_type: str | None = None,
        background: BackgroundTask | None = None,
    ) -> None:
        if isinstance(content, AsyncIterable):
            self._content_iterable: AsyncIterable = content
        else:
            self._content_iterable = iterate_in_threadpool(content)

        async def body_iterator() -> AsyncIterable[bytes]:
            async for content_ in self._content_iterable:
                if isinstance(content_, BaseModel):
                    content_ = content_.model_dump()
                yield self.render(content_)

        self.body_iterator = body_iterator()
        self.status_code = status_code
        if media_type is not None:
            self.media_type = media_type
        self.background = background
        self.init_headers(headers)

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            cls=LangChainEncoder,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")

#
###
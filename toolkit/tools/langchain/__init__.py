###
#
from fastapi import APIRouter

import os
import time
import rich
import inspect

from typing import Mapping, Union, Optional, Any
from collections.abc import AsyncIterator, Iterator
from enum import Enum
from pydantic import BaseModel, Field
#from langchain.pydantic_v1 import BaseModel, Field

from ...utils import JSONStreamingResponse

#
###

###
#
import jinja2
from pathlib import Path

current_path = Path(__file__).parent.absolute()
TEMPLATE_DIR = os.environ.get("LANGCHAIN_TEMPLATE_DIR", "templates")
TEMPLATE_PATH = os.path.join(current_path, TEMPLATE_DIR)

TEMPLATE_LOADER = jinja2.FileSystemLoader(searchpath=TEMPLATE_PATH)
TEMPLATE_ENV = jinja2.Environment(loader=TEMPLATE_LOADER)

DEFAULT_SYSTEM_PROMPT = """
You are SYNTHESIS. You respond to user input however you see fit.
Be imaginative but keep your response focused on the user prompt.
""".lstrip().rstrip()

#
###

###
#
# langchain
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, MessagesPlaceholder

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "0.0.0.0")
OLLAMA_PORT = os.environ.get("OLLAMA_PORT", "11434")

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "dolphin-llama3:8b-256k-v2.9-q8_0")
OLLAMA_SMALL = os.environ.get("OLLAMA_SMALL", "tinydolphin:1.1b-v2.8-q8_0")
OLLAMA_LARGE = os.environ.get("OLLAMA_LARGE", "dolphin-llama3:70b-v2.9-q6_K ")
OLLAMA_EMBED = os.environ.get("OLLAMA_EMBED", "mxbai-embed-large:latest")
OLLAMA_CODE = os.environ.get("OLLAMA_CODE", "codestral:22b-v0.1-q8_0")

class ModelTypes(Enum):
    MODEL = OLLAMA_MODEL
    SMALL = OLLAMA_SMALL
    LARGE = OLLAMA_LARGE
    EMBED = OLLAMA_EMBED
    CODE = OLLAMA_CODE

class ClientTypes(Enum):
    CHAT = "chat"
    EMBED = "embed"
    TOOL = "tool"

CLIENT_MAP = {
    "chat":ChatOllama,
    "embed":OllamaEmbeddings,
    "tool":OllamaFunctions,
    }

#
# TODO
# - rework ModelTypes => how?
#
def gen_client(kind: Optional[ClientTypes] = None, host: Optional[str] = None, port: Optional[str] = None, model: Optional[str] = None, **kwargs):
    kind = kind or ClientTypes.CHAT
    if not isinstance(kind, ClientTypes):
        kind = ClientTypes(kind.lower())

    host = host or OLLAMA_HOST
    port = port or OLLAMA_PORT
    model = model or ModelTypes.MODEL.value

    base_url = f"http://{host}:{port}"

    rich.print(f"KWARGS: {kwargs}")
    return CLIENT_MAP[kind.value](model=model, base_url=base_url, **kwargs)

#
###


###
#
LangChainRouter = APIRouter()
@LangChainRouter.get("/")
async def root() -> Mapping[str, Union[int, str]]:
    return {}


#
###

###
#
class ChatRequest(BaseModel):
    model_type: ModelTypes = ModelTypes.MODEL
    model_kind: ClientTypes = ClientTypes.CHAT
    host: Optional[str] = None
    port: Optional[str] = None
    client_kwargs: Optional[dict[str,str]] = {}

    system_template_file: Optional[str] = "chat_template.jinja2"

    system_prompt: Optional[str] = None or DEFAULT_SYSTEM_PROMPT
    context_list: Optional[list[dict[str,str]]] = None
    history_list: Optional[list[dict[str,str]]] = None
    user_prompt: Optional[dict[str, str]] = None
    assistant_prompt: Optional[dict[str, str]] = None

    stream: bool = False

    class Config:
        protected_namespaces = ()

@LangChainRouter.post("/chat", response_model=None)
async def chat(data: ChatRequest) -> Union[Mapping[str, Any], AsyncIterator[Mapping[str, Any]]]:
    for k in inspect.signature(gen_client).parameters.keys():
        data.client_kwargs.pop(k, None)

    llm = gen_client(kind=data.model_kind, host=data.host, port=data.port, model=data.model_type.value, **data.client_kwargs)

    t = TEMPLATE_ENV.get_template(data.system_template_file).render()
    system_prompt = SystemMessagePromptTemplate.from_template(t, template_format="jinja2")
    c = {}
    for k in system_prompt.input_variables:
        if v := getattr(data, k):
            c[k] = v
    system_prompt = system_prompt.format_messages(**c)

    history_list = []
    for history in data.history_list:
        rich.print(f"HISTORY - {history}")
        if history["role"] == "user":
            history_list.append(("human", history["content"]))

        else:
            history_list.append(("ai", history["content"]))

    message_list = system_prompt + history_list + [("human", "{user_prompt}")]

    prompt = ChatPromptTemplate.from_messages(message_list)

    chain = prompt | llm | StrOutputParser()

    prompt_kwargs = {}
    prompt_kwargs["user_prompt"] = data.user_prompt["content"]

    if data.stream:
        result = chain.astream_events(input=prompt_kwargs, version="v2")

        async def _gen(g):
            async for r in g:
                rich.print(r)
                yield r

        return JSONStreamingResponse(_gen(result), media_type="application/json")

    else:
        return await chain.ainvoke(input=prompt_kwargs)

#
###

###
#
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from langchain.pydantic_v1 import BaseModel as V1BaseModel
from langchain.pydantic_v1 import Field as V1Field

class ToolKind(Enum):
    langchain = "langchain"
    code = "code"
    plugin = "plugin"
    mongo = "mongo"

class ToolObject(BaseModel):
    name: str
    kind: Optional[ToolKind] = ToolKind.langchain
    metadata: dict[str, Any] = {}

class AgentRequest(ChatRequest):
    model_kind: ClientTypes = ClientTypes.TOOL
    tool_list: list[ToolObject] = []

class InputResponse(BaseModel):
    key: str
    data: dict[str, Any] = {}

@LangChainRouter.post("/agent", response_model=None)
async def agent(data: AgentRequest):
    raise NotImplementedError()
    for k in inspect.signature(gen_client).parameters.keys():
        data.client_kwargs.pop(k, None)

    llm = gen_client(
        kind=data.model_kind,
        host=data.host,
        port=data.port,
        model=data.model_type.value,
        **data.client_kwargs)




@LangChainRouter.post("/tool_input", response_model=None)
async def tool_input(data: InputResponse):
    raise NotImplementedError()

#
###
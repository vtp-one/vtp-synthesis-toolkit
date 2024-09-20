from typing import Type, Optional, Any
from langchain.pydantic_v1 import BaseModel as V1BaseModel
from langchain.pydantic_v1 import Field as V1Field

from langchain.tools import BaseTool, StructuredTool, tool

from os.path import dirname, basename, isfile, join
import glob
import importlib
from copy import copy
modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.startswith("_")]


SYNTHESIS_TOOLS = {}
for z in __all__:
    if z.endswith("_tool"):
        q = copy(z)
        q = list(q.removesuffix("_tool"))
        q[0] = q[0].upper()
        q = "".join(q)
        q += "Tool"

        #importlib.import_module(f".{z}", __name__)
        #__import__(f"{z}.{q}")
        #SYNTHESIS_TOOLS[z] = eval(f"{z}.{q}._gen_tool")



from langchain.agents.load_tools import get_all_tool_names
def get_tool_names(synthesis_only: bool = True):
    if synthesis_only:
        return list(SYNTHESIS_TOOLS.keys())

    else:
        return get_all_tool_names() + list(SYNTHESIS_TOOLS.keys())

from langchain.agents.load_tools import (
    load_tools,
    _handle_callbacks,
    DANGEROUS_TOOLS,
    _BASE_TOOLS,
    _LLM_TOOLS,
    _EXTRA_LLM_TOOLS,
    _EXTRA_OPTIONAL_TOOLS,
    )

from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import Callbacks
from langchain_core.tools import BaseTool
import inspect

def load_tools(
    tool_names: list[str],
    llm: Optional[BaseLanguageModel] = None,
    callbacks: Callbacks = None,
    allow_dangerous_tools: bool = True,
    **kwargs: Any,
    ) -> list[BaseTool]:

    tools = []
    callbacks = _handle_callbacks(
        callback_manager=kwargs.get("callback_manager"), callbacks=callbacks
    )
    for name in tool_names:
        if name in SYNTHESIS_TOOLS.keys():
            _tool = SYNTHESIS_TOOLS[name]
            _sig = inspect.signature(_tool)
            if len(_sig.parameters.keys()):
                _kwargs = {}
                for p in _sig.parameters.items():
                    if p == "llm":
                        _kwargs["llm"] = llm

                    else:
                        _kwargs[p] = kwargs.pop(p)

                tools.append(_tool(**kwargs))

            else:
                tools.append(_tool())

            continue

        if name in DANGEROUS_TOOLS and not allow_dangerous_tools:
            raise ValueError(
                f"{name} is a dangerous tool. You cannot use it without opting in "
                "by setting allow_dangerous_tools to True. "
                "Most tools have some inherit risk to them merely because they are "
                'allowed to interact with the "real world".'
                "Please refer to LangChain security guidelines "
                "to https://python.langchain.com/docs/security."
                "Some tools have been designated as dangerous because they pose "
                "risk that is not intuitively obvious. For example, a tool that "
                "allows an agent to make requests to the web, can also be used "
                "to make requests to a server that is only accessible from the "
                "server hosting the code."
                "Again, all tools carry some risk, and it's your responsibility to "
                "understand which tools you're using and the risks associated with "
                "them."
            )

        if name in {"requests"}:
            warnings.warn(
                "tool name `requests` is deprecated - "
                "please use `requests_all` or specify the requests method"
            )
        if name == "requests_all":
            # expand requests into various methods
            requests_method_tools = [
                _tool for _tool in _BASE_TOOLS if _tool.startswith("requests_")
            ]
            tool_names.extend(requests_method_tools)
        elif name in _BASE_TOOLS:
            tools.append(_BASE_TOOLS[name]())
        elif name in DANGEROUS_TOOLS:
            tools.append(DANGEROUS_TOOLS[name]())
        elif name in _LLM_TOOLS:
            if llm is None:
                raise ValueError(f"Tool {name} requires an LLM to be provided")
            tool = _LLM_TOOLS[name](llm)
            tools.append(tool)
        elif name in _EXTRA_LLM_TOOLS:
            if llm is None:
                raise ValueError(f"Tool {name} requires an LLM to be provided")
            _get_llm_tool_func, extra_keys = _EXTRA_LLM_TOOLS[name]
            missing_keys = set(extra_keys).difference(kwargs)
            if missing_keys:
                raise ValueError(
                    f"Tool {name} requires some parameters that were not "
                    f"provided: {missing_keys}"
                )
            sub_kwargs = {k: kwargs[k] for k in extra_keys}
            tool = _get_llm_tool_func(llm=llm, **sub_kwargs)
            tools.append(tool)
        elif name in _EXTRA_OPTIONAL_TOOLS:
            _get_tool_func, extra_keys = _EXTRA_OPTIONAL_TOOLS[name]
            sub_kwargs = {k: kwargs[k] for k in extra_keys if k in kwargs}
            tool = _get_tool_func(**sub_kwargs)
            tools.append(tool)
        else:
            raise ValueError(f"Got unknown tool {name}")
    if callbacks is not None:
        for tool in tools:
            tool.callbacks = callbacks
    return tools

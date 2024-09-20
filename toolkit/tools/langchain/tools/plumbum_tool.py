import rich
from typing import Type, Optional, Annotated
from langchain.pydantic_v1 import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from langchain.tools import BaseTool, StructuredTool, tool

import plumbum
import shlex

class PlumbumInput(BaseModel):
    cmd: str = Field(description="The command to run")

class PlumbumTool(BaseTool):
    name = "Call Plumbum"
    description = "Use Plumbum to call system executables with provided arguments"
    args_schema: Type[BaseModel] = PlumbumInput

    def _run(self, cmd, run_manager: Optional[CallbackManagerForToolRun] = None):
        cmd = shlex.split(cmd)
        z = plumbum.machines.LocalCommand(cmd[0])
        q = z.popen(args=cmd[1:])
        g = plumbum.commands.base.iter_lines(q)

        accumulator = []
        try:
            for l in g:
                rich.print(l)

                if run_manager:
                    run_manager.on_text(l)

                accumulator.append(l)
                #
                # yield {tool, content}
                #

        except Exception as exc:
            rich.print(exc)

            #
            # yield {tool, error}
            #

        return accumulator


    async def _arun(self, cmd, run_manager: Optional[AsyncCallbackManagerForToolRun] = None):
        #
        # TODO
        # - plumbum async
        #
        cmd = shlex.split(cmd)
        z = plumbum.machines.LocalCommand(cmd[0])
        q = z.popen(args=cmd[1:])
        g = plumbum.commands.base.iter_lines(q)

        accumulator = []
        try:
            for l in g:
                rich.print(l)

                if run_manager:
                    await run_manager.on_text(l)

                accumulator.append(l)
                #
                # yield {tool, content}
                #

        except Exception as exc:
            rich.print(exc)

            #
            # yield {tool, error}
            #

        return accumulator

    @classmethod
    def _gen_tool(cls):
        return cls()

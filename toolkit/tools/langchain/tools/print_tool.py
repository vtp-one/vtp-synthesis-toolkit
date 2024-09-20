import rich
from rich.rule import Rule
from typing import Type, Optional, Annotated, Any
from langchain.pydantic_v1 import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from langchain.tools import BaseTool, StructuredTool, tool

class PrintInput(BaseModel):
    content: Any = Field(description="The content to print")

class PrintTool(BaseTool):
    name = "Print Tool"
    description = "Print the input content to the terminal"
    args_schema: Type[BaseModel] = PrintInput

    def _print(self, content):
        rich.print(Rule())
        rich.print("PrintTool")
        rich.print(content)
        rich.print(Rule())

    def _run(self, content, run_manager: Optional[CallbackManagerForToolRun] = None) -> Any:
        self._print(content)
        return content

    async def _arun(self, content, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> Any:
        self._print(content)
        return content

    @classmethod
    def _gen_tool(cls):
        return cls()

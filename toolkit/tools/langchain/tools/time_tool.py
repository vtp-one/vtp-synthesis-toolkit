import rich
import time
from rich.rule import Rule
from typing import Type, Optional, Annotated, Any, Literal
from langchain.pydantic_v1 import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from langchain.tools import BaseTool, StructuredTool, tool

class TimeInput(BaseModel):
    format_: Literal["default", "asctime"] = Field(default="default", description="The format of timestamp to return")

class TimeTool(BaseTool):
    name = "Time Tool"
    description = "Return the time in a specific format"
    args_schema: Type[BaseModel] = TimeInput

    def _run(self, format_, run_manager: Optional[CallbackManagerForToolRun] = None) -> Any:
        match format_:
            case "asctime":
                return f"The current time in asc format is: {time.asctime()}"

            case _:
                return f"The current time in default format is: {time.time()}"

    async def _arun(self, format_, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> Any:
        return self.run(format_, run_manager)

    @classmethod
    def _gen_tool(cls):
        return cls()

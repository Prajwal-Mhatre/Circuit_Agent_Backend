import json
from pathlib import Path
from typing import Optional
from llama_index.program.openai import OpenAIPydanticProgram
from pydantic import BaseModel, field_validator

# --- STRUCTURED OUTPUT SCHEMAS ---
class ProjectComponent(BaseModel):
    component_name: str
    quantity: int
    description: str

class ProjectComponentList(BaseModel):
    """Structured schema for generating a project name and component list."""
    project_name: str
    components: list[ProjectComponent]

# --- MAIN CLASS ---
class Agent1:
    def __init__(self, llm, output_path: str = "agent1_output.json"):
        self.llm = llm
        self.output_path = Path(output_path)
        self.prompt_template = (
            "You are an expert hardware engineer.\n"
            "Given the following user request, generate a structured JSON object with:\n"
            "- project_name (string)\n"
            "- components (a list of objects with fields: component_name, quantity, description)\n\n"
            "1. you only need to mention electronic componenets required to build the device and can skip components like jumper wire or more."
            "2. The component list must include exactly one microcontroller.\n"
            "   - Choose the microcontroller from this list: ['Arduino_Uno']\n"
            "3. Do not add any extra microcontrollers. Use only one from the list above.\n\n"
            "User request: {user_query}"
        )

        self.program = OpenAIPydanticProgram.from_defaults(
            output_cls=ProjectComponentList,
            prompt_template_str=self.prompt_template,
            llm=self.llm,
            verbose=True,
        )


    def run(self, user_query: str, save: bool = True) -> ProjectComponentList:
        result = self.program(user_query=user_query)
        if save:
            with open(self.output_path, "w") as f:
                json.dump(result.model_dump(), f, indent=2)
        self.previous_output = result
        return result



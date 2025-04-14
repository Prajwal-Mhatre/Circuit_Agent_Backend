import json
from pathlib import Path
from typing import Optional
from llama_index.program.openai import OpenAIPydanticProgram
from pydantic import BaseModel, field_validator

# --- SCHEMAS ---
class ProjectComponent(BaseModel):
    component_name: str
    quantity: int
    description: str

class ProjectComponentList(BaseModel):
    project_name: str
    components: list[ProjectComponent]

# --- TRACKER CLASS ---
class ComponentGenerator:
    def __init__(self, llm, output_path: str = "component_output.json"):
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

        self.previous_output: Optional[ProjectComponentList] = self._load_previous_output()

    def _load_previous_output(self) -> Optional[ProjectComponentList]:
        if self.output_path.exists():
            with open(self.output_path, "r") as f:
                data = json.load(f)
                return ProjectComponentList(**data)
        return None

    def run(self, user_query: str, save: bool = True) -> ProjectComponentList:
        output = self.program(user_query=user_query)
        if save:
            with open(self.output_path, "w") as f:
                json.dump(output.model_dump(), f, indent=2)
        self.previous_output = output
        return output

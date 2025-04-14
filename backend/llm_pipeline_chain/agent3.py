import json
from pathlib import Path
from typing import List, Optional, Union
from llama_index.program.openai import OpenAIPydanticProgram
from pydantic import BaseModel


# --- SCHEMAS ---
class ComponentCode(BaseModel):
    component: str
    code: str

class FullCodeOutput(BaseModel):
    """ code for each component to make the embedded project work"""
    project: str
    code_blocks: List[ComponentCode]


# --- MAIN CLASS ---
class Agent3:
    def __init__(self, llm, output_path: str = "agent3_output.json", previous_specs: Optional[List[dict]] = None,):
        self.llm = llm
        self.output_path = Path(output_path)
        self.previous_specs = previous_specs or []

        self.prompt_template = (
            "You are an expert embedded systems developer.\n"
            "You are provided with a JSON hardware specification: which is this = \n{hardware_spec}\n , it includes:\n"
            "- project: the name of the project\n"
            "- description: what the project is intended to do\n"
            "- components: the hardware components used, including microcontrollers and sensors\n"
            "- connections: how components are connected via their pins\n\n"
            "Use the description to understand the project's behavior, and the components + connections to generate correct code.\n"
            "For each programmable component (like microcontrollers or sensors), write compatible code that performs its role in the project.\n\n"
            "Skip passive or non-programmable parts like resistors, capacitors, or breadboards.\n\n"
            "Return a JSON object with the following structure:\n"
            "- project: name of the project\n"
            "- code_blocks: a list where each item includes:\n"
            "    - component: name of the programmable component\n"
            "    - code: working Arduino code that controls or interacts with that component\n\n"
            "Make the code clean, minimal, and logically aligned with the purpose described."
        )

        self.program = OpenAIPydanticProgram.from_defaults(
            output_cls=FullCodeOutput,
            prompt_template_str=self.prompt_template,
            llm=self.llm,
            description="Generate code for each programmable hardware component.",
            verbose=True,
        )



    def run(self, hardware_spec: Union[dict, BaseModel], save: bool = True) -> FullCodeOutput:
        if isinstance(hardware_spec, BaseModel):
            hardware_spec = hardware_spec.model_dump()

        input_str = json.dumps(hardware_spec, indent=2)
        result = self.program(hardware_spec=input_str)

        if save:
            with open(self.output_path, "w") as f:
                json.dump(result.model_dump(), f, indent=2)

        print(json.dumps(result.model_dump(), indent=2))
        return result



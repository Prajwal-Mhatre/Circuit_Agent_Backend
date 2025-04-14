import json
from pathlib import Path
from typing import List, Optional, Union
from llama_index.program.openai import OpenAIPydanticProgram
from pydantic import BaseModel, field_validator, Field
from backend.pin_map_svg_generator.svg_generator import main

# --- STRUCTURED OUTPUT SCHEMAS ---
class Connection(BaseModel):
    from_: List[str] = Field(..., min_items=2, max_items=2)
    to: List[str]= Field(..., min_items=2, max_items=2)
    @field_validator("from_", "to")
    @classmethod
    def validate_two_strings(cls, v, field):
        if len(v) != 2:
            raise ValueError(f"'{field.name}' must have exactly 2 elements")
        if not all(isinstance(item, str) for item in v):
            raise ValueError(f"All items in '{field.name}' must be strings")
        return v

class Component(BaseModel):
    name: str
    pins: List[str]
    type: str
    name: str
    pins: List[str]
    type: str  # keep it str so we can map everything, even custom values

    @field_validator("type", mode="before")
    @classmethod
    def enforce_component_type(cls, value: str) -> str:
        allowed_types = {"resistor", "switch", "relay", "motor", "power_supply", "battery"}
        value = value.strip().lower()
        return value if value in allowed_types else "custom_IC"
    
class FullHardwareSpec(BaseModel):
    """Detailed hardware layout including pin mapping and connections."""
    project: str
    description: str
    components: List[Component]
    connections: List[Connection]



#component_data = json.dumps(output.model_dump(), indent=2)

# --- MAIN CLASS ---
class Agent2:
    def __init__(self,llm,output_path: str = "hardware_pin_map.json",previous_component_outputs: Optional[List[dict]] = None):
        self.llm = llm
        self.output_path = Path(output_path)
        self.previous_component_outputs = previous_component_outputs or []

        self.prompt_template = (
            "You are a hardware system designer.\n"
            "Given this component list:\n{component_data}\n\n"
            "Generate a detailed JSON structure with:\n"
            "- project: name of the project\n"
            "- description: a short, clear description of what the device does strictly in the context of this project\n"
            "- components: list of components used, each with:\n"
            "    - name\n"
            "    - pins (list of pin names)\n"
            "    - type: one of the following strictly — resistor, switch, relay, motor, battery, custom_IC\n"
            "- connections: list of connections between [device, pin] to [device, pin]\n\n"
            "Only include relevant electronic components, and use the project description to guide what pins or devices may be needed.\n"
            #    "You MUST only use arduino_uno_pins = [ D0, D1, D2, D3, D4, D5, D6, D7,D8, D9, D10, D11, D12, D13, A0, A1, A2, A3, A4, A5, VIN, 5V, 3.3V, GND, GND, GND,RESET, IOREF, AREF]\n"
            #    "for pins of Ardunio_Uno component if present"
            #    "Only use this accurate pin map of Arduino_Uno is :arduino_uno_pins = [ D0, D1, D2, D3, D4, D5, D6, D7,D8, D9, D10, D11, D12, D13, A0, A1, A2, A3, A4, A5, VIN, 5V, 3.3V, GND, GND, GND,RESET, IOREF, AREF]"
        )

        self.program = OpenAIPydanticProgram.from_defaults(
            output_cls=FullHardwareSpec,
            prompt_template_str=self.prompt_template,
            llm=self.llm,
            verbose=True,
            description="Generate detailed pin connections and component layout from base component list.",
        )




    def run(self, component_data, save: bool = True) -> FullHardwareSpec:
        component_json_str = json.dumps(component_data, indent=2)
        #component_json_str = component_data
        result = self.program(component_data=component_json_str)

        if save:
            with open(self.output_path, "w") as f:
                json.dump(result.model_dump(), f, indent=2)

        # Pass to drawing function  (run the svg generator)
        main(result.model_dump()) 


        return result
    

   


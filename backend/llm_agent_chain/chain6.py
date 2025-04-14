from pydantic import BaseModel, field_validator
from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.llms.openai import OpenAI
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
import tiktoken
from dotenv import load_dotenv
load_dotenv()
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 
from pin_map_svg_generator.svg_generator import main
from cost_tracker.cost_tracker import CostTracker
from typing import List, Union
import json


# 1. Define the output schema
class ProjectComponent(BaseModel):
    component_name: str
    quantity: int
    description: str

class ProjectComponentList(BaseModel):
    """Structured schema for generating a project name and component list."""
    project_name: str
    components: list[ProjectComponent]



class Connection(BaseModel):
    from_: List[str]  # use from_ because `from` is a Python keyword
    to: List[str]

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
        allowed_types = {"resistor", "switch", "relay", "motor", "power_supply"}
        value = value.strip().lower()
        if value in allowed_types:
            return value
        return "custom_IC"
    
class FullHardwareSpec(BaseModel):
    """Detailed hardware layout including pin mapping and connections."""
    project: str
    description: str
    components: List[Component]
    connections: List[Connection]

# 2. Prompt template that takes a user query
prompt_template_str = (
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

# we need to add a api call that gives the component specs from the database.

# Initialize the TokenCountingHandler
token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-4o-mini").encode
)
# Set up the CallbackManager
callback_manager = CallbackManager([token_counter])

# 3. Setup LLM (ensure it's GPT-4o-mini)
llm = OpenAI(model="gpt-4o-mini",callback_manager=callback_manager)
#llm = OpenAI(model="gpt-4o-mini-2024-07-18")


# 4. Create the program with a user-query placeholder
program = OpenAIPydanticProgram.from_defaults(
    output_cls=ProjectComponentList,
    prompt_template_str=prompt_template_str,
    llm=llm,
    verbose=True,
)

# 5. Run it using the user's natural language query
user_query = "Build me a plant watering system."
output = program(user_query=user_query)

# 6. Output is structured and validated by Pydantic
#print(output)
print("program 1 generated")

# Save to file
import json
with open("component_output.json", "w") as f:
    json.dump(output.model_dump(), f, indent=2)


# Convert output to JSON string
component_data = json.dumps(output.model_dump(), indent=2)

second_prompt = (
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

  
second_program = OpenAIPydanticProgram.from_defaults(
    output_cls=FullHardwareSpec,
    prompt_template_str=second_prompt,
    llm=llm,
    description="Generate detailed pin connections and component layout from base component list.",
    verbose=True,
)

# Run second model
detailed_spec = second_program(component_data=component_data)

# Save to file
with open("hardware_pin_map.json", "w") as f:
    json.dump(detailed_spec.model_dump(), f, indent=2)

# Print if needed
# print(detailed_spec)
print("program 2 generated")

# Pass to drawing function
main(detailed_spec.model_dump())


#############################################################################
## new program to start code

# Output schema for code blocks
class ComponentCode(BaseModel):
    component: str
    code: str

class FullCodeOutput(BaseModel):
    """ code for each component to make the embedded project work"""
    project: str
    code_blocks: List[ComponentCode]

# Program 3 prompt
code_gen_prompt = (
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

program3 = OpenAIPydanticProgram.from_defaults(
    output_cls=FullCodeOutput,
    prompt_template_str=code_gen_prompt,
    llm=llm,
    description="Generate code for each programmable hardware component.",
    verbose=True,
)

#detailed_spec = json.dumps(detailed_spec.model_dump(), indent=2)

# Run Program 3
program3_output = program3(hardware_spec=json.dumps(detailed_spec.model_dump(), indent=2))

# Save to file
with open("component_code_output.json", "w") as f:
    json.dump(program3_output.model_dump(), f, indent=2)

# Print it
print(json.dumps(program3_output.model_dump(), indent=2))





###calculate cost

# Prices (per 1K tokens)
input_price_per_1k = 0.00015  # $0.15 per million input tokens
output_price_per_1k = 0.00060  # $0.60 per million output tokens

# Token usage
prompt_tokens = token_counter.prompt_llm_token_count
completion_tokens = token_counter.completion_llm_token_count
total_tokens = token_counter.total_llm_token_count

# Cost calculation
input_cost = (prompt_tokens / 1000) * input_price_per_1k
output_cost = (completion_tokens / 1000) * output_price_per_1k
total_cost = input_cost + output_cost

# Print report
print("\n🧾 Token Usage Summary:")
print(f"Prompt tokens     : {prompt_tokens}")
print(f"Completion tokens : {completion_tokens}")
print(f"Total tokens      : {total_tokens}")

print("\n💸 Estimated Cost Breakdown (USD):")
print(f"Input  cost       : ${input_cost:.6f}")
print(f"Output cost       : ${output_cost:.6f}")
print(f"Total  cost       : ${total_cost:.6f}")

tracker = CostTracker()  # or CostTracker("my_custom_path.json")
# After your cost calc
tracker.add_cost(input_cost, output_cost, total_cost)
#append to total file
print(tracker.total_cost)


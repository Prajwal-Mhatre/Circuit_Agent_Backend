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

def load_components(filename="components.json"):
    base_dir = os.path.dirname(__file__)  # directory of the current .py file
    path = os.path.join(base_dir, "..", filename)  # go one level up
    with open(path, "r") as f:
        return json.load(f)

COMPONENTS = load_components()

# ✅ STEP 2: Function to expose to LLM
from typing import TypedDict, List

def normalize(name: str) -> str:
    return name.lower().replace(" ", "_")

class ComponentInfo(TypedDict):
    pins: List[str]
    type: str

def get_component_info(name: str) -> ComponentInfo:
    print(f"🔧 Tool called for: {name}")
    key = normalize(name)
    result = COMPONENTS.get(key, {"pins": ["UNKNOWN"], "type": "custom_IC"})
    if result:
        return result
    else:
        return {
            "pins": [],
            "type": "not_found"
        }            


# ✅ STEP 3: Create LlamaIndex tool
from llama_index.core.tools import FunctionTool
component_tool = FunctionTool.from_defaults(
    fn=get_component_info,
    name="get_component_info",
    description="Fetches pins and type for a given component name."
)

# 1. Define the output schema
class ProjectComponent(BaseModel):
    component_name: str
    quantity: int
    type: str

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
    components: List[Component]
    connections: List[Connection]

# # 2. Prompt template that takes a user query
# prompt_template_str = (
#     "You are an expert hardware engineer.\n"
#     "Given the following user request, generate a structured JSON object with:\n"
#     "- project_name (string)\n"
#     "- components (a list of objects with fields: component_name, quantity, description)\n\n"
#     "1. you only need to mention electronic componenets required to build the device and can skip components like jumper wire or more."
#     "2. The component list must include exactly one microcontroller.\n"
#     "   - Choose the microcontroller from this list: ['Arduino_Uno']\n"
#     "3. Do not add any extra microcontrollers. Use only one from the list above.\n\n"
#     "User request: {user_query}"
# )

prompt_template_str = (
    "You are a hardware engineer designing a new project.\n"
    "Step 1: Think about which components are needed.\n"
    "Step 2: For each component you need, call the `get_available_components` tool.\n"
    "Use only components that exist in the database. If the tool returns type='not_found', do not include it.\n"
    "Return the final JSON structure listing only valid components, each with their pins and type.\n\n"
    "User request: {user_query}"
)

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
    tools=[component_tool],  # 👈 Tool is registered here
    llm=llm,
    verbose=True,
)

# 5. Run it using the user's natural language query
user_query = "Build me a plant watering system."
output = program()

# 6. Output is structured and validated by Pydantic
print(output)

# Save to file
import json
with open("component_output.json", "w") as f:
    json.dump(output.model_dump(), f, indent=2)



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



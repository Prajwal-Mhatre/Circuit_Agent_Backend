from llama_index.core.program import FunctionCallingProgram
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel

class Song(BaseModel):
    name: str
    artist: str

prompt_template_str = "Generate a song about (topic)."
llm = OpenAI(model="gpt-3.5-turbo")

program = FunctionCallingProgram.from_defaults(
    output_cls=Song,
    prompt_template_str=prompt_template_str,
    llm=llm,
    verbose=True,
)

output = program(topic="space exploration")
print(output)

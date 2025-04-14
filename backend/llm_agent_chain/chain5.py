
from pydantic import BaseModel
from typing import List
from llama_index.llms.openai import OpenAI
from llama_index.program.openai import OpenAIPydanticProgram
from dotenv import load_dotenv
load_dotenv()
import os


class Song(BaseModel):
    title: str
    length_seconds: int


class Album(BaseModel):
    name: str
    artist: str
    songs: List[Song]


prompt_template_str = """\
Generate an example album, with an artist and a list of songs. \
Using the movie {movie_name} as inspiration.\
"""
program = OpenAIPydanticProgram.from_defaults(
    output_cls=Album, llm = OpenAI(model="gpt-4o-mini"), prompt_template_str=prompt_template_str, verbose=True
)

output = program(
    movie_name="The Shining", description="Data model for an album."
)

###so this is a working code.



from pydantic import BaseModel
from llama_index.core.query_engine import RetrieverQueryEngine
# Define a Pydantic model for structured output
class QueryResponse(BaseModel):
    answer: str
    sources: list
# Modify your query function to use structured output
def query(self, your_query):
    '''
    This function requires a LLM and its API key.
    It uses the LLM API to generate output with RAG knowledge.
    '''
    # Create a query engine with structured output
    query_engine = self.index.as_query_engine(output_cls=QueryResponse)
    # Query the index
    response = query_engine.query(your_query)
    # Print the structured response
    print(response.json(indent=2))
# Example usage
# Assuming `self.index` is already set up and contains your indexed data
your_query = "What is the capital of France?"
query(your_query)
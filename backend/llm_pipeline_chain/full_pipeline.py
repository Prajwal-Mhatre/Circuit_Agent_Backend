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
from cost_tracker.token_counter import TokenCounter
from llm_pipeline_chain.agent1 import Agent1
from llm_pipeline_chain.agent2 import Agent2
from llm_pipeline_chain.agent3 import Agent3

from typing import List, Union
import json


#setup token counter+ cost calculator
token_counter = TokenCounter("got-4o-mini")

# 3. Setup LLM (ensure it's GPT-4o-mini)
llm = OpenAI(model="gpt-4o-mini",callback_manager=token_counter.get_call_back_manager())
#llm = OpenAI(model="gpt-4o-mini-2024-07-18")

user_query = "build be a fast flying drone"

agent1 = Agent1(llm)
output1 = agent1.run(user_query)

# we need to add a api call that gives the component specs from the database.

agent2 = Agent2(llm)
output2 = agent2.run(output1)

agent3 = Agent3(llm)
output3 = agent3.run(output2)


###calculate cost
# Token usage and cost calculator
token_counter.calculate_cost_till_now()
# Print usage of this session
token_counter.print_usage()

#Print all usage of your api
token_counter.print_total_usage_till_now()


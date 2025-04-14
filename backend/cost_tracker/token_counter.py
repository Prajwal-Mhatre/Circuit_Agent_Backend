from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
import tiktoken
from ..cost_tracker import CostTracker

class TokenCounter:
    def __init__(self,llm_model):
        
        # Initialize the TokenCountingHandler
        self.token_counter = TokenCountingHandler(
            tokenizer=tiktoken.encoding_for_model(llm_model).encode
        )

        # Set up the CallbackManager
        self.call_back_manager = CallbackManager([self.token_counter])

        # Prices (per 1K tokens)
        self.input_price_per_1k = 0.00015  # $0.15 per million input tokens
        self.output_price_per_1k = 0.00060  # $0.60 per million output tokens

        # create the tracker to track and store to file.
        self.tracker = CostTracker()  # or CostTracker("my_custom_path.json")

    def get_call_back_manager(self):
        return self.call_back_manager
    
    def track_and_store_usage(self):

        self.calculate_cost_till_now()

        # After your cost calc
        self.tracker.add_cost(self.input_cost, self.output_cost, self.total_cost)

    def calculate_cost_till_now(self):
        # Token usage
        self.prompt_tokens = self.token_counter.prompt_llm_token_count
        self.completion_tokens = self.token_counter.completion_llm_token_count
        self.total_tokens = self.token_counter.total_llm_token_count

        # Cost calculation
        self.input_cost = (self.prompt_tokens / 1000) * self.input_price_per_1k
        self.output_cost = (self.completion_tokens / 1000) * self.output_price_per_1k
        self.total_cost = self.input_cost + self.output_cost

    def print_usage(self):
        # Print report
        print("\n🧾 Token Usage Summary:")
        print(f"Prompt tokens     : {self.prompt_tokens}")
        print(f"Completion tokens : {self.completion_tokens}")
        print(f"Total tokens      : {self.total_tokens}")

        print("\n💸 Estimated Cost Breakdown (USD):")
        print(f"Input  cost       : ${self.input_cost:.6f}")
        print(f"Output cost       : ${self.output_cost:.6f}")
        print(f"Total  cost       : ${self.total_cost:.6f}")

    def print_total_usage_till_now(self):
        print(self.tracker.total_cost)    
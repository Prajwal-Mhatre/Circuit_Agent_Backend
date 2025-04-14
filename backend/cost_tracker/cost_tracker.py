import json
import os

class CostTracker:
    def __init__(self, path="cost_data.json"):
        self.path = path
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                data = json.load(f)
        else:
            data = {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
        
        self.input_cost = data["input_cost"]
        self.output_cost = data["output_cost"]
        self.total_cost = data["total_cost"]

    def add_cost(self, input_cost: float, output_cost: float, total_cost: float):
        self.input_cost += input_cost
        self.output_cost += output_cost
        self.total_cost += total_cost
        self._save()

    def _save(self):
        with open(self.path, "w") as f:
            json.dump({
                "input_cost": round(self.input_cost, 6),
                "output_cost": round(self.output_cost, 6),
                "total_cost": round(self.total_cost, 6)
            }, f, indent=2)

            
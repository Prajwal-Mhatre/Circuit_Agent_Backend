import json
import os
from typing import List, Dict, Any, Optional
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from openai import OpenAI as OpenAIClient

# Load environment variables
import dotenv
dotenv.load_dotenv()

# Configure the OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# Initialize OpenAI client
openai_client = OpenAIClient(api_key=OPENAI_API_KEY)

# Set up LlamaIndex with GPT-4o Mini
llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
Settings.llm = llm

class EmbeddedProjectAssistant:
    def __init__(self, components_json_path: str):
        """Initialize the assistant with a components inventory."""
        self.components = self._load_components(components_json_path)
        # Create embeddings for components for better matching
        self._create_component_index()

    def _load_components(self, json_path: str) -> Dict[str, Any]:
        """Load the components inventory from a JSON file."""
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Components file not found at {json_path}. Creating a new one.")
            # Create a default structure for components
            default_components = {"components": []}
            with open(json_path, 'w') as f:
                json.dump(default_components, f, indent=2)
            return default_components
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {json_path}")

    def _create_component_index(self):
        """Create a vector index of components for similarity matching."""
        # Extract component details as documents
        documents = []
        for component in self.components.get("components", []):
            doc_text = f"Name: {component.get('name', '')}\n"
            doc_text += f"Type: {component.get('type', '')}\n"
            doc_text += f"Description: {component.get('description', '')}\n"
            doc_text += f"Specifications: {component.get('specifications', '')}\n"
            documents.append(doc_text)
        
        if documents:
                            # Create index from documents
            from llama_index.core.schema import Document
            docs = [Document(text=text) for text in documents]
            self.component_index = VectorStoreIndex.from_documents(docs)
        else:
            self.component_index = None
            print("No components found to index.")

    def analyze_project_query(self, user_query: str) -> Dict[str, Any]:
        """
        Analyze the user's project query to determine required components.
        
        Args:
            user_query: The user's description of the embedded project
            
        Returns:
            A dictionary with the project analysis results
        """
        # Define the function schema for component identification
        functions = [
            {
                "name": "identify_required_components",
                "description": "Identify the components required for an embedded project",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_name": {
                            "type": "string",
                            "description": "Name of the embedded project"
                        },
                        "project_description": {
                            "type": "string",
                            "description": "Brief description of what the project does"
                        },
                        "required_components": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "Name of the required component"
                                    },
                                    "type": {
                                        "type": "string",
                                        "description": "Type of component (e.g., microcontroller, sensor, display, etc.)"
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "Description of the component's role in the project"
                                    },
                                    "specifications": {
                                        "type": "string",
                                        "description": "Any specific requirements or specifications for this component"
                                    },
                                    "critical": {
                                        "type": "boolean",
                                        "description": "Whether this component is critical for the project's functionality"
                                    }
                                },
                                "required": ["name", "type", "description"]
                            },
                            "description": "List of components required for the project"
                        }
                    },
                    "required": ["project_name", "project_description", "required_components"]
                }
            }
        ]

        # Prompt the LLM to analyze the project query
        prompt = f"""
        You are an embedded systems expert. Analyze the following project description and 
        identify all components that would be required to build it.
        
        For each component, determine:
        1. Its name
        2. The type of component (e.g., microcontroller, sensor, display)
        3. A description of its role in the project
        4. Any specific specifications it needs to meet
        5. Whether it's critical for the project's functionality
        
        Project description: {user_query}
        """

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            functions=functions,
            function_call={"name": "identify_required_components"}
        )

        # Extract the function call arguments
        function_call = response.choices[0].message.function_call
        if function_call and function_call.name == "identify_required_components":
            components_data = json.loads(function_call.arguments)
            return components_data
        else:
            raise ValueError("Failed to analyze project requirements")

    def check_component_availability(self, required_components: List[Dict[str, Any]]) -> Dict[str, List]:
        """
        Check which components are available in inventory and suggest alternatives for missing ones.
        
        Args:
            required_components: List of components required for the project
            
        Returns:
            Dictionary with available and missing components, along with alternatives
        """
        available_components = []
        missing_components = []
        
        inventory = self.components.get("components", [])
        
        for required in required_components:
            found = False
            for inventory_item in inventory:
                # Check if the component matches by name (case-insensitive)
                if required["name"].lower() == inventory_item.get("name", "").lower():
                    # Check if it meets all specifications
                    specs_match = True
                    if "specifications" in required and required["specifications"]:
                        # Simple check - in real-world scenario, this would be more sophisticated
                        if required["specifications"].lower() not in inventory_item.get("specifications", "").lower():
                            specs_match = False
                    
                    if specs_match:
                        available_components.append({
                            "required": required,
                            "inventory_item": inventory_item
                        })
                        found = True
                        break
            
            if not found:
                missing_components.append(required)
        
        # For missing components, find alternatives
        alternatives = self._find_alternatives(missing_components)
        
        return {
            "available": available_components,
            "missing": missing_components,
            "alternatives": alternatives
        }

    def _find_alternatives(self, missing_components: List[Dict[str, Any]]) -> Dict[str, List]:
        """
        Find alternative components for missing ones.
        
        Args:
            missing_components: List of components that are missing from inventory
            
        Returns:
            Dictionary mapping missing component names to lists of alternatives
        """
        alternatives = {}
        inventory = self.components.get("components", [])
        
        if not inventory or not missing_components:
            return alternatives
            
        for missing in missing_components:
            missing_name = missing.get("name", "")
            missing_type = missing.get("type", "")
            missing_specs = missing.get("specifications", "")
            
            # Combine information for a better query
            query = f"Component name: {missing_name}. Type: {missing_type}. Specifications: {missing_specs}"
            
            if self.component_index:
                # Use vector search to find similar components
                query_engine = self.component_index.as_query_engine()
                results = query_engine.query(query)
                
                # Get component names from the results
                component_alternatives = []
                
                # Extract component names from the search results
                for node in results.source_nodes:
                    for component in inventory:
                        # Check if this component is in the node text
                        if component.get("name", "").lower() in node.text.lower():
                            component_alternatives.append(component)
                            break
                
                if component_alternatives:
                    alternatives[missing_name] = component_alternatives
                else:
                    # Fallback to function calling with LLM
                    alternatives[missing_name] = self._find_alternatives_with_llm(missing, inventory)
            else:
                # Fallback to function calling with LLM
                alternatives[missing_name] = self._find_alternatives_with_llm(missing, inventory)
                
        return alternatives

    def _find_alternatives_with_llm(self, missing_component: Dict[str, Any], inventory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find alternatives using the LLM's function calling capability."""
        # Define the function schema for finding alternatives
        functions = [
            {
                "name": "find_component_alternatives",
                "description": "Find alternative components that could substitute for a missing component",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "alternatives": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "inventory_id": {
                                        "type": "integer",
                                        "description": "Index of the component in the inventory list"
                                    },
                                    "compatibility_score": {
                                        "type": "integer",
                                        "description": "Score from 1-10 indicating how well this component can substitute for the missing one"
                                    },
                                    "adaptation_notes": {
                                        "type": "string",
                                        "description": "Notes on any adaptations needed to use this component"
                                    }
                                },
                                "required": ["inventory_id", "compatibility_score"]
                            },
                            "description": "List of potential alternative components"
                        }
                    },
                    "required": ["alternatives"]
                }
            }
        ]

        # Format the inventory for the prompt
        inventory_text = ""
        for i, item in enumerate(inventory):
            inventory_text += f"[{i}] Name: {item.get('name', '')}, "
            inventory_text += f"Type: {item.get('type', '')}, "
            inventory_text += f"Description: {item.get('description', '')}, "
            inventory_text += f"Specifications: {item.get('specifications', '')}\n"

        # Prompt the LLM to find alternatives
        prompt = f"""
        You are an embedded systems expert. A project requires the following component which is not available:
        
        Name: {missing_component.get('name', '')}
        Type: {missing_component.get('type', '')}
        Description: {missing_component.get('description', '')}
        Specifications: {missing_component.get('specifications', '')}
        
        Here is the inventory of available components:
        
        {inventory_text}
        
        Find components from the inventory that could be used as alternatives. Return their inventory
        index, a compatibility score (1-10), and notes on any adaptations needed.
        
        If no suitable alternatives exist, return an empty list.
        """

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            functions=functions,
            function_call={"name": "find_component_alternatives"}
        )

        # Extract the function call arguments
        function_call = response.choices[0].message.function_call
        if function_call and function_call.name == "find_component_alternatives":
            alternatives_data = json.loads(function_call.arguments)
            
            # Map the alternatives to the actual inventory items
            result = []
            for alt in alternatives_data.get("alternatives", []):
                inventory_id = alt.get("inventory_id")
                if 0 <= inventory_id < len(inventory):
                    inventory_item = inventory[inventory_id].copy()
                    inventory_item["compatibility_score"] = alt.get("compatibility_score")
                    inventory_item["adaptation_notes"] = alt.get("adaptation_notes", "")
                    result.append(inventory_item)
            
            return result
        else:
            return []

    def process_user_query(self, user_query: str) -> Dict[str, Any]:
        """
        Process a user query about an embedded project.
        
        Args:
            user_query: The user's description of the embedded project
            
        Returns:
            A dictionary with the analysis results
        """
        # Step a1: Analyze the project query to identify required components
        project_analysis = self.analyze_project_query(user_query)
        
        # Step 2: Check component availability and find alternatives
        availability = self.check_component_availability(project_analysis.get("required_components", []))
        
        # Combine the results
        results = {
            "project_name": project_analysis.get("project_name", ""),
            "project_description": project_analysis.get("project_description", ""),
            "components": {
                "available": availability.get("available", []),
                "missing": availability.get("missing", []),
                "alternatives": availability.get("alternatives", {})
            }
        }
        
        return results

    def generate_project_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable report of the project analysis.
        
        Args:
            results: The results from process_user_query
            
        Returns:
            A formatted report string
        """
        report = f"# Project Report: {results['project_name']}\n\n"
        report += f"## Project Description\n{results['project_description']}\n\n"
        
        # Available components
        report += "## Available Components\n"
        if results['components']['available']:
            for item in results['components']['available']:
                required = item['required']
                inventory = item['inventory_item']
                report += f"- **{required['name']}** ({required['type']}): {required['description']}\n"
                report += f"  - Inventory match: {inventory['name']}\n"
        else:
            report += "_No components available in inventory._\n"
        
        report += "\n"
        
        # Missing components
        report += "## Missing Components\n"
        if results['components']['missing']:
            for missing in results['components']['missing']:
                report += f"- **{missing['name']}** ({missing['type']}): {missing['description']}\n"
                
                # Add alternatives if any
                alternatives = results['components']['alternatives'].get(missing['name'], [])
                if alternatives:
                    report += "  - **Possible alternatives:**\n"
                    for alt in alternatives:
                        report += f"    - {alt['name']} (Compatibility: {alt.get('compatibility_score', 'N/A')}/10)\n"
                        if alt.get('adaptation_notes'):
                            report += f"      - _Note: {alt['adaptation_notes']}_\n"
                else:
                    report += "  - **No suitable alternatives found**\n"
        else:
            report += "_All required components are available._\n"
        
        return report


# Example usage
if __name__ == "__main__":
    # Path to your components inventory JSON file
    components_json_path = "components_inventory.json"
    
    # Initialize the assistant
    assistant = EmbeddedProjectAssistant(components_json_path)
    
    # Example user query
    user_query = "I want to build a weather station that measures temperature, humidity, and barometric pressure. It should display the readings on an LCD screen and upload the data to a cloud service every 10 minutes."
    
    # Process the query
    results = assistant.process_user_query(user_query)
    
    # Generate and print a report
    report = assistant.generate_project_report(results)
    print(report)

# Example components_inventory.json structure:
"""
{
  "components": [
    {
      "name": "ESP32",
      "type": "microcontroller",
      "description": "Dual-core microcontroller with WiFi and Bluetooth",
      "specifications": "240MHz, 520KB SRAM, WiFi, BLE"
    },
    {
      "name": "DHT22",
      "type": "sensor",
      "description": "Temperature and humidity sensor",
      "specifications": "Temperature range: -40°C to 80°C, Humidity range: 0-100% RH"
    },
    {
      "name": "BMP280",
      "type": "sensor",
      "description": "Barometric pressure sensor",
      "specifications": "Pressure range: 300-1100 hPa"
    },
    {
      "name": "LCD1602",
      "type": "display",
      "description": "16x2 character LCD display",
      "specifications": "16 columns, 2 rows, I2C interface"
    }
  ]
}
"""
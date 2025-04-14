import json

# Step 1: Load the original component code JSON
with open("component_code_output.json", "r") as f:
    data = json.load(f)

# Step 2: Create a plain text output
output_lines = []

output_lines.append(f"Project: {data.get('project', 'Unnamed Project')}\n")

for block in data.get("code_blocks", []):
    component = block.get("component", "Unknown")
    raw_code = block.get("code", "")

    # Decode \n, \t, etc.
    pretty_code = raw_code.encode('utf-8').decode('unicode_escape')

    output_lines.append(f"\nComponent: {component}")
    output_lines.append("-" * 40)
    output_lines.append(pretty_code)
    output_lines.append("\n")  # spacing

# Step 3: Save to a readable .txt file
with open("component_code_output_printed_code.txt", "w") as f:
    f.write("\n".join(output_lines))



from flask import Flask, request, jsonify, send_file
import requests

app = Flask(__name__)

OPENAI_API_KEY = 'your-openai-key-here'

# 👇 now only the code blocks are fetched in step 3
BACKEND_CODE_API = 'http://localhost:5022/generate-code'
BACKEND_SVG_API = 'http://localhost:5050/process'  # accessed directly by frontend

@app.route('/')
def serve_index():
    return send_file('index.html')

@app.route('/app.js')
def serve_js():
    return send_file('app.js')

@app.route('/api/chat', methods=['POST'])
def chat():
    user_input = request.json.get("input", "")
    
    # Step 1: Get components + full spec + code blocks
    backend_response = requests.post(
        f"{BACKEND_API}/generate-all",
        json={"input": user_input}
    )
    
    if backend_response.status_code != 200:
        return jsonify({"error": "Backend failed", "details": backend_response.text}), 500

    result = backend_response.json()

    return jsonify({
        "step1": result.get("component_list", {}).get("project_name", "No project"),
        "step2": result.get("full_spec", {}),
        "step3": result.get("code_blocks", {}),
        "projectData": result.get("full_spec", {})  # used by JS for SVG drawing
    })

if __name__ == '__main__':
    app.run(port=5001, debug=True)


# @app.route('/api/chat', methods=['POST'])
# def chat():
#     structured_json = {
#       "project": "Flame Throwing Robot",  ##
#       "components": [
#           {
#               "name": "Arduino_Uno",
#               "pins": ["D2", "D3", "D4", "D5", "D6", "D7"]
#           },
#           {
#               "name": "Flame Thrower",
#               "pins": ["Power", "Control"]
#           },
#           {
#               "name": "Servo Motor",
#               "pins": ["D8", "D9"]
#           }
#       ],
#       "connections": [
#           {
#               "from_": ["Arduino_Uno", "D2"],
#               "to": ["Servo Motor", "D8"]
#           },
#           {
#               "from_": ["Arduino_Uno", "D3"],
#               "to": ["Flame Thrower", "Control"]
#           }
#       ]
#     }

#     return jsonify({
#         "step1": "Hardcoded structure",
#         "step2": structured_json,
#         "step3": {},
#         "projectData": structured_json  # used by frontend JS to POST to /process
#     })

# if __name__ == '__main__':
#     app.run(port=5001, debug=True)

from flask import Flask, request, send_file, after_this_request
print("🧠 backend_app.py is running")
from flask_cors import CORS
from llm_pipeline_chain.agent1 import Agent1
from llm_pipeline_chain.agent2 import Agent2
from llm_pipeline_chain.agent3 import Agent3
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
from pin_map_svg_generator.svg_generator import main  # Adjust import to match your real filename/function
import os,json

app = Flask(__name__)
CORS(app)  # ✅ This allows all origins by default

@app.route('/process', methods=['POST'])
def process():
    print("[BACKEND] /process was hit")
    data = request.get_json()
    print("[🛠️ BACKEND] Received JSON:", data)

    try:
        svg_path = main(data)
        print("[✅ BACKEND] SVG saved at:", svg_path)
    except Exception as e:
        print("[❌ BACKEND] Error in SVG generation:", e)
        return "SVG generation failed", 500
    

    @after_this_request
    def cleanup(response):
        try:
            #os.remove(svg_path)
            print("[🧹 BACKEND] Temp SVG deleted")
        except Exception as e:
            print("[⚠️ BACKEND] Failed to delete SVG:", e)            
            print(f"Failed to delete file: {e}")
        return response

    return send_file(svg_path, mimetype='image/svg+xml')



if __name__ == '__main__':
    app.run(port=5050, debug=True)


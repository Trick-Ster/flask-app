# Import necessary libraries
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from huggingface_hub import login

# Authenticate with Hugging Face if token is available
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    login(token=hf_token)

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer from Hugging Face
model_name = "mistralai/Mistral-Nemo-Instruct-2407"  # Replace with your chosen model
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    model, tokenizer = None, None

@app.route('/process', methods=['POST'])
def process_query():
    """
    This route accepts a query (text) via a POST request,
    processes it using the NeMo model, and returns the result.
    """
    if model is None or tokenizer is None:
        return jsonify({"error": "Model is not loaded. Please check your configuration."}), 500

    # Get the query from the incoming JSON payload
    data = request.get_json()
    query = data.get("query", "")

    if query:
        try:
            # Tokenize the input query
            inputs = tokenizer(query, return_tensors="pt")

            # Perform inference using the model
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=50)

            # Decode the generated response
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Return the response as a JSON object
            return jsonify({
                "query": query,
                "response": response_text
            })
        except Exception as e:
            return jsonify({"error": f"An error occurred while processing the query: {e}"}), 500
    else:
        # Return error if no query was provided
        return jsonify({"error": "No query provided"}), 400

# Start the Flask app
if __name__ == '__main__':
    # Use 0.0.0.0 to ensure the app runs in Docker/Render environments
    app.run(host="0.0.0.0", port=5000, debug=True)

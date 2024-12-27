# Import necessary libraries
from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os  # Import os to access environment variables

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer from Hugging Face
model_name = "mistralai/Mistral-Nemo-Instruct-2407"  # You can replace this with another NeMo model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

@app.route('/process', methods=['POST'])
def process_query():
    """
    This route will accept a query (text) via a POST request,
    process it using the NeMo model, and return the result.
    """
    # Get the query from the incoming JSON payload
    data = request.get_json()
    query = data.get("query", "")

    if query:
        # Tokenize the input query
        inputs = tokenizer(query, return_tensors="pt")

        # Perform inference using the model
        with torch.no_grad():
            outputs = model(**inputs)

        # Example: Extract prediction or logits (you can modify this based on your model)
        prediction = outputs.logits.argmax(dim=-1).item()  # Taking the class with max logit

        # Return the prediction as a JSON response
        return jsonify({
            "query": query,
            "prediction": prediction  # Return the predicted class index (or another result)
        })
    else:
        # Return error if no query was provided
        return jsonify({"error": "No query provided"}), 400

# Start the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))  # Use PORT environment variable or default to 5000

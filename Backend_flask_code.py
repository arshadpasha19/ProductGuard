from flask import Flask, request, jsonify, send_file
import json
import numpy as np
from ingredients_analysis_model import process_image
from skin_type_model import predict_skin_type
from recommender_model import get_overall_recommendations
import os
import openai

#creates a Flask application instance
app = Flask(__name__)

# Route to handle image processing and pie chart generation
@app.route('/ingredient_analysis', methods=['POST'])
def generate_ingredients_dataset():
    try:
        # Check if the request contains an image file
        if 'image' not in request.files:
            return "No image provided in the request", 400

        # Get the image file from the request
        image_file = request.files['image']

        # Determine which temporary image file to save based on request data
        if request.form.get('productId') == '1':
            image_path = "temp_image1.jpg"
        elif request.form.get('productId') == '2':
            image_path = "temp_image2.jpg"
        else:
            image_path = "temp_image.jpg"

        # Save the image file to the appropriate temporary location
        image_file.save(image_path)

        # Call the function to process the image and generate the pie chart
        df = process_image(image_path)

        # Convert DataFrame to JSON
        df_json = df.to_json(orient='records')

        return jsonify(df_json)

    except Exception as e:
        return jsonify({"error": str(e)})

# Route to handle image classification
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided."})

        image_file = request.files['image']
        image_data = image_file.read()
        pred_probabilities = predict_skin_type(image_data)

        return jsonify({"probabilities": {
            "dry": pred_probabilities[0],
            "normal": pred_probabilities[1],
            "oily": pred_probabilities[2]
        }})

    except Exception as e:
        return jsonify({"error": str(e)})

# Route to handle recommendation
@app.route('/recommend', methods=['POST'])
def recommend_products():
    try:
        # Extract user input values from the request
        data = request.json
        concerns = data.get('concerns')

        # Pass user input values to the recommender model
        recommendations = get_overall_recommendations(list(concerns.values()))

        return jsonify({"products": recommendations})

    except Exception as e:
        return jsonify({"error": str(e)})

# Set your OpenAI API key
openai.api_key = "YOUR OPENAI API KEY"

def get_chat_response(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

# Route to handle Chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        # Get the user's query from the request
        user_query = request.json['query']

        # Get the response from the OpenAI ChatBot
        answer = get_chat_response(user_query)
        # Return the response to the client
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

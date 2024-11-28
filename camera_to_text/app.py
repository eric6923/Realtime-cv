import os
from flask import Flask, render_template, request, jsonify
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
import google.generativeai as genai

app = Flask(__name__)

# Set your Azure subscription key and endpoint
subscription_key = "EpsWFbmeMkd7Q5xeWDKNzbY7mVrcGAJfZ2fB4Lwc5hh6sTYJq0X6JQQJ99AKACYeBjFXJ3w3AAAFACOGpiT3"
endpoint = "https://csvdemo.cognitiveservices.azure.com/"

# Create an Image Analysis client
client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(subscription_key))

# Configure the Gemini API
genai.configure(api_key="AIzaSyCAI14HgK1SJd_jh4XxeVEoSh5Bgf8ydM4")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    image = request.files['file']
    if not image:
        return jsonify({'error': 'Invalid image file.'}), 400

    try:
        # Read image data
        image_data = image.read()
        
        # Send image to Azure for OCR
        result = client.analyze(image_data=image_data, visual_features=[VisualFeatures.READ])
        
        # Process the results
        if result.read is not None:
            text = ""
            for line in result.read.blocks[0].lines:
                text += line.text + " "
            return jsonify({'text': text.strip()})
        else:
            return jsonify({'error': 'No text found in the image'}), 500

    except Exception as e:
        print(f"Exception occurred: {str(e)}")  # Log the exception
        return jsonify({'error': 'Error processing image.'}), 500

@app.route('/gemini', methods=['POST'])
def gemini():
    user_input = request.json.get('prompt')
    
    # Modify the prompt to request a concise answer
    modified_prompt = f"Provide a short answer {user_input}"

    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(modified_prompt)

    if response:
        chat_response = response.text
        return jsonify({"response": chat_response})
    else:
        return jsonify({"error": "Failed to get a response from Gemini API"}), 500

if __name__ == '__main__':
    app.run(debug=True)

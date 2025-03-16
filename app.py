from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from datetime import timedelta
from google import genai
from google.genai import types
import os
import json
import ast

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(hours=1)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
CORS(app)


def generate(image):
    try:
        #put API generate from google ai studio in api_credentials.json
        with open("api_credentials.json", "r") as f:
            credentials = json.load(f)
            api_key = credentials["api_key"]
    except FileNotFoundError:
        raise ValueError("API credentials file not found.")
    except KeyError:
        raise ValueError("API key not found in credentials file.")

    client = genai.Client(
        api_key=api_key,
    )

    files = [
        # Make the file available in local system working directory
        client.files.upload(file=image),
    ]
    model = "gemini-2.0-flash-lite"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=files[0].uri,
                    mime_type=files[0].mime_type,
                ),
                types.Part.from_text(
                    #text to generate desired output
                    text="""interprate the image showing medical skin lesion and produce 5 probable diagnosis in 
                    python dictionary format in one line [{'diagnosis, VALUE', 'probability, VALUE'}, ...(continue)] without disclaimer.
                    highest to lowest probability"""
                ),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_mime_type="application/json", #this will give output as python string in dictionary structure
    )

    full_response = "" #collect all chunks of the response.
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        #print(chunk.text, end="")
        full_response += chunk.text #append text to the response

    print(full_response)
    return full_response #return the entire response.


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/images')
def images():
    return render_template('images.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'error, No file part'

    file = request.files['file']

    if file.filename == '':
        return 'error, No selected file'

    # Handle the file as needed (e.g., save it, process it)
    print('file uploaded')
    return 'message, File uploaded successfully'


@app.route('/detect', methods=['GET','POST'])
def detect():
    # Check if the POST request has the file part
    if 'image' not in request.files:
        return 'error, No file part'

    image_data = request.files['image']

    # If the user does not select a file, submit an empty part without filename
    if image_data.filename == '':
        return 'error, No selected file'
    
    # Process the uploaded image
    
    if image_data:
        filename = image_data.filename
        #temp_path = os.path.join('/tmp', filename) #linux
        temp_path = os.path.join('tmp', filename) #windows
        image_data.save(temp_path)

        #generate gemini output
        dict_output = generate(temp_path)

        #strip and remove unnecessary strings
        dict_output = dict_output.strip().strip("```")
        dict_output = dict_output.replace("python", "")

        #convert to python dictionary
        dict_output = ast.literal_eval(dict_output)

        #convert probability value to float enable it to convert percentage
        for d in dict_output:
            d['probability'] = round(float(d['probability'])*100, 2) #limit percentage to 2 decimal

        print(dict_output)

        return render_template('detect.html',
                            dict_output = dict_output,)
    else:
        return "status, process error"



if __name__ == "__main__":
    app.run(debug=True,host='127.0.0.1',port=int(os.environ.get('PORT', 5000)))

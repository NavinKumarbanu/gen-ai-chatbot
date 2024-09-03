from flask import Flask, render_template, request, jsonify
import vertexai
from vertexai.language_models import ChatModel, TextGenerationModel

app = Flask(__name__)

# Initialize Vertex AI
vertexai.init(project="reflected-codex-434416-p6", location="us-central1")

# Define parameters for the model
parameters = {
    "candidate_count": 1,
    "max_output_tokens": 64,
    "temperature": 1,
    "top_p": 0.8,
    "top_k": 40
}

# Load the text generation model
model = TextGenerationModel.from_pretrained("text-unicorn@001")

# Create a session with the chat model
def create_session():
    chat_model = ChatModel.from_pretrained("chat-bison@001")
    return chat_model

# Generate a response from the model
def response(chat_model, message):
    chat = chat_model.start_chat()
    result = chat.send_message(message, **parameters)
    return result.text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/palm2', methods=['GET', 'POST'])
def vertex_palm():
    user_input = ""
    if request.method == 'GET':
        user_input = request.args.get('user_input')
    else:
        user_input = request.form['user_input']
    
    # Create chat model session and generate a response
    chat_model = create_session()
    content = response(chat_model, user_input)
    return jsonify(content=content)

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')


from flask import Flask, render_template, request, jsonify
from chatbot import generate_response  # Import chatbot function

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")  # Load UI

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    bot_reply = generate_response(user_message)  # Get chatbot response
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)



# from flask import Flask, render_template, request, jsonify
# from chatbot import generate_response  # Import chatbot function

# app = Flask(__name__)

# @app.route("/")
# def index():
#     return render_template("index.html")  # Load UI

# @app.route("/chat", methods=["POST"])
# def chat():
#     user_message = request.json.get("message")
#     bot_reply = generate_response(user_message)  # Get chatbot response
#     return jsonify({"reply": bot_reply})

# if __name__ == "__main__":
#     app.run(debug=True)
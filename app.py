from flask import Flask, render_template, request
from chatbot import *

bot = chatbotAI()
app = Flask(__name__, template_folder="templates")


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET","POST"])
def chat():
    msg = request.form["msg"]
    print(msg)
    return bot.chatAPI(msg)

if __name__ == '__main__':
    app.run()

from flask import Flask
app = Flask(__name__)

@app.route('/getStatus')
def hello_world():
    return 'Server is running!'
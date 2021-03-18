from flask import request, jsonify
from flask_lambda import FlaskLambda

app = FlaskLambda(__name__)

@app.route("/validate_email", methods=("POST",))
def validate_email():
    data = request.get_json()

    return jsonify({"data": data}), 200
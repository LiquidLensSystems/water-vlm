from flask import Flask, render_template, jsonify
import requests
import sys
import os
import time
import threading
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

app = Flask(__name__)

# =========================================================
# ROUTES
# =========================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/sat_pos")
def sat_pos():
    if os.path.exists("satellite_state.json"):
        with open("satellite_state.json") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Syncing..."}), 404

@app.route('/stream')
def stream():
    if os.path.exists("shared_state.json"):
        with open("shared_state.json", "r") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Syncing..."}), 404


if __name__ == '__main__':
    app.run(port=5000, debug=True)

# remote_server.py

from flask import Flask, request, jsonify
import subprocess
import threading

app = Flask(__name__)

# Path to the program you want to run on this remote machine:
PROGRAM_PATH = "part2.py"


@app.route('/trigger', methods=['POST'])
def trigger():
    subprocess.run(["python3", PROGRAM_PATH])
    return jsonify({"status": "launched"}), 200

if __name__ == "__main__":
    # Listen on all interfaces so your Pi can reach it
    app.run(host="0.0.0.0", port=5001)

from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    # Local: http://127.0.0.1:5000
    # Network: http://YOUR_LOCAL_IP:5000 (devices on the same network)
    app.run(host="0.0.0.0", port=5000, debug=True)

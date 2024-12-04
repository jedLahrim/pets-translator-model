from app import app  # Import the Flask app from your app.py file

if __name__ == "__main__":
    app.run(debug=False, host="localhost", port=5002)

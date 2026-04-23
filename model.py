from tensorflow.keras.models import load_model
import numpy as np

model = load_model("lstm_model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    sequence = data["sequence"]  # list of last hours

    X = np.array([sequence])

    prob = model.predict(X)[0][0]
    risk = round(prob * 100, 2)

    if risk < 30:
        status = "Normal"
    elif risk < 70:
        status = "Warning"
    else:
        status = "High Risk"

    return jsonify({
        "risk": risk,
        "status": status
    })
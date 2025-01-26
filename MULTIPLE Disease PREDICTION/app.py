# from flask import Flask, request, render_template
# import joblib
# import numpy as np

# app = Flask(__name__)

# # Load the trained model
# model = joblib.load("models/heart.pkl")

# @app.route("/")
# def home():
#     return render_template("home.html")


# @app.route("/heart", methods=['GET', 'POST'])
# def heart():
#     if request.method == "POST":
#         # Collect all the form data
#         age = float(request.form["age"])
#         sex = float(request.form["sex"])
#         cp = float(request.form["cp"])
#         trestbps = float(request.form["trestbps"])
#         chol = float(request.form["chol"])
#         fbs = float(request.form["fbs"])
#         restecg = float(request.form["restecg"])
#         thalach = float(request.form["thalach"])
#         exang = float(request.form["exang"])
#         oldpeak = float(request.form["oldpeak"])
#         slope = float(request.form["slope"])
#         ca = float(request.form["ca"])
#         thal = float(request.form["thal"])

#         # Create a NumPy array of the input features (order must match the training features)
#         input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
#                                 exang, oldpeak, slope, ca, thal]])

#         # Predict using the trained model
#         prediction = model.predict(input_data)

#         # Output the prediction result
#         result = "Positive for Heart Disease" if prediction[0] == 1 else "Negative for Heart Disease"
        
#         return render_template("heart.html", prediction=result)

#     return render_template("heart.html")

# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
heart_model = joblib.load("models/heart.pkl")
diabetes_model = joblib.load("models/diabetes.pkl")
parkinsons_model = joblib.load("models/parkinsons.pkl")
cancer_model = joblib.load("models/cancer.pkl")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/heart", methods=['GET', 'POST'])
def heart():
    input_data = {}
    prediction = None

    if request.method == "POST":
        # Collect all the form data
        input_data = {
            "age": request.form.get("age", ""),
            "sex": request.form.get("sex", ""),
            "cp": request.form.get("cp", ""),
            "trestbps": request.form.get("trestbps", ""),
            "chol": request.form.get("chol", ""),
            "fbs": request.form.get("fbs", ""),
            "restecg": request.form.get("restecg", ""),
            "thalach": request.form.get("thalach", ""),
            "exang": request.form.get("exang", ""),
            "oldpeak": request.form.get("oldpeak", ""),
            "slope": request.form.get("slope", ""),
            "ca": request.form.get("ca", ""),
            "thal": request.form.get("thal", ""),
        }

        try:
            # Convert input to NumPy array for prediction
            input_array = np.array([[float(input_data[key]) for key in input_data]])
            prediction = heart_model.predict(input_array)

            # Determine the prediction result
            result = "Positive for Heart Disease" if prediction[0] == 1 else "Negative for Heart Disease"
        except ValueError:
            result = "Invalid input. Please enter valid numerical values."

        return render_template("heart.html", prediction=result, input_data=input_data)

    return render_template("heart.html", input_data=input_data)


@app.route("/diabetes", methods=['GET', 'POST'])
def diabetes():
    input_data = {}
    prediction = None

    if request.method == "POST":
        # Collect all the form data for diabetes
        input_data = {
            "pregnancies": request.form.get("pregnancies", ""),
            "glucose": request.form.get("glucose", ""),
            "blood_pressure": request.form.get("blood_pressure", ""),
            "skin_thickness": request.form.get("skin_thickness", ""),
            "insulin": request.form.get("insulin", ""),
            "bmi": request.form.get("bmi", ""),
            "diabetes_pedigree_function": request.form.get("diabetes_pedigree_function", ""),
            "age": request.form.get("age", ""),
        }

        try:
            # Convert input to NumPy array for prediction
            input_array = np.array([[float(input_data[key]) for key in input_data]])

            # Predict using the diabetes model
            prediction = diabetes_model.predict(input_array)

            # Determine the prediction result
            result = "Positive for Diabetes" if prediction[0] == 1 else "Negative for Diabetes"
        except ValueError:
            result = "Invalid input. Please enter valid numerical values."

        return render_template("diabetes.html", prediction=result, input_data=input_data)

    return render_template("diabetes.html", input_data=input_data)


@app.route("/cancer", methods=['GET', 'POST'])
def cancer():
    input_data = {}
    prediction = None

    if request.method == "POST":
        # Collect all the form data
        input_data = {
            "radius_mean": request.form.get("radius_mean", ""),
            "texture_mean": request.form.get("texture_mean", ""),
            "perimeter_mean": request.form.get("perimeter_mean", ""),
            "area_mean": request.form.get("area_mean", ""),
            "smoothness_mean": request.form.get("smoothness_mean", ""),
            "compactness_mean": request.form.get("compactness_mean", ""),
            "concavity_mean": request.form.get("concavity_mean", ""),
            "concave_points_mean": request.form.get("concave_points_mean", ""),
            "symmetry_mean": request.form.get("symmetry_mean", ""),
            "fractal_dimension_mean": request.form.get("fractal_dimension_mean", ""),
            "radius_se": request.form.get("radius_se", ""),
            "texture_se": request.form.get("texture_se", ""),
            "perimeter_se": request.form.get("perimeter_se", ""),
            "area_se": request.form.get("area_se", ""),
            "smoothness_se": request.form.get("smoothness_se", ""),
            "compactness_se": request.form.get("compactness_se", ""),
            "concavity_se": request.form.get("concavity_se", ""),
            "concave_points_se": request.form.get("concave_points_se", ""),
            "symmetry_se": request.form.get("symmetry_se", ""),
            "fractal_dimension_se": request.form.get("fractal_dimension_se", ""),
            "radius_worst": request.form.get("radius_worst", ""),
            "texture_worst": request.form.get("texture_worst", ""),
            "perimeter_worst": request.form.get("perimeter_worst", ""),
            "area_worst": request.form.get("area_worst", ""),
            "smoothness_worst": request.form.get("smoothness_worst", ""),
            "compactness_worst": request.form.get("compactness_worst", ""),
            "concavity_worst": request.form.get("concavity_worst", ""),
            "concave_points_worst": request.form.get("concave_points_worst", ""),
            "symmetry_worst": request.form.get("symmetry_worst", ""),
            "fractal_dimension_worst": request.form.get("fractal_dimension_worst", ""),
        }

        try:
            # Convert input to NumPy array for prediction
            input_array = np.array([[float(input_data[key]) for key in input_data]])
            
            # Make the prediction using the trained model
            prediction = cancer_model.predict(input_array)

            # Determine the prediction result
            result = "Positive for Cancer" if prediction[0] == 1 else "Negative for Cancer"
        except ValueError:
            result = "Invalid input. Please enter valid numerical values."

        return render_template("cancer.html", prediction=result, input_data=input_data)

    return render_template("cancer.html", input_data=input_data)


@app.route("/parkinsons", methods=['GET', 'POST'])
def parkinsons():
    input_data = {}
    prediction = None

    if request.method == "POST":
        # Collect all the form data
        input_data = {
            "name": request.form.get("name", ""),
            "MDVP_Fo_Hz": request.form.get("MDVP_Fo_Hz", ""),
            "MDVP_Fhi_Hz": request.form.get("MDVP_Fhi_Hz", ""),
            "MDVP_Flo_Hz": request.form.get("MDVP_Flo_Hz", ""),
            "MDVP_Jitter_percent": request.form.get("MDVP_Jitter_percent", ""),
            "MDVP_Jitter_Abs": request.form.get("MDVP_Jitter_Abs", ""),
            "MDVP_RAP": request.form.get("MDVP_RAP", ""),
            "MDVP_PPQ": request.form.get("MDVP_PPQ", ""),
            "Jitter_DDP": request.form.get("Jitter_DDP", ""),
            "MDVP_Shim": request.form.get("MDVP_Shim", ""),
            "MDVP_Shim_dB": request.form.get("MDVP_Shim_dB", ""),
            "Shimmer_APQ3": request.form.get("Shimmer_APQ3", ""),
            "Shimmer_APQ5": request.form.get("Shimmer_APQ5", ""),
            "MDVP_APQ": request.form.get("MDVP_APQ", ""),
            "Shimmer_DDA": request.form.get("Shimmer_DDA", ""),
            "NHR": request.form.get("NHR", ""),
            "HNR": request.form.get("HNR", ""),
            "RPDE": request.form.get("RPDE", ""),
            "DFA": request.form.get("DFA", ""),
            "spread1": request.form.get("spread1", ""),
            "spread2": request.form.get("spread2", ""),
            "D2": request.form.get("D2", ""),
            "PPE": request.form.get("PPE", ""),
        }

        try:
            # Convert input to NumPy array for prediction
            input_array = np.array([[float(input_data[key]) for key in input_data if key != "name"]])
            
            # Make the prediction using the trained model
            prediction = parkinsons_model.predict(input_array)

            # Determine the prediction result
            result = "Positive for Parkinson's Disease" if prediction[0] == 1 else "Negative for Parkinson's Disease"
        except ValueError:
            result = "Invalid input. Please enter valid numerical values."

        return render_template("parkinsons.html", prediction=result, input_data=input_data)

    return render_template("parkinsons.html", input_data=input_data)



if __name__ == "__main__":
    app.run(debug=True)

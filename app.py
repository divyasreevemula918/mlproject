# from flask import Flask,request,render_template
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from src.pipeline.predict_pipeline import CustomData,PredictPipeline
# application=Flask(__name__)
# app=application
# @app.route('/')
# def home():
#     return render_template('index.html')
# # @app.route('/predictdata',methods=['GET','POST'])
# # def predict_datapoint():
# #     if request.method=='GET':
# #         return render_template('home.html')
# #     else:
# #         data=CustomData(
# #             gender=request.form.get('gender'),
# #             race_ethnicity=request.form.get('race_ethnicity'),
# #             parental_level_of_education=request.form.get('parental_level_of_education'),
# #             lunch=request.form.get('lunch'),
# #             test_preparation_course=request.form.get('test_preparation_course'),
# #             reading_score=float(request.form.get('reading_score')),
# #             writing_score=float(request.form.get('writing_score'))
# #         )
# #         pred_df=data.get_data_as_dataframe()
# #         print(pred_df)
# #         Predict_Pipeline=PredictPipeline()
# #         results=Predict_Pipeline.predict(pred_df)
# #         return render_template('home.html',results=results[0])
# # if __name__=="__main__":
# #     app.run(host='0.0.0.0',debug=True)
# @app.route('/predictdata',methods=['GET','POST'])
# def predict_datapoint():
#     if request.method=='GET':
#         return render_template('index.html')   # FIXED
#     else:
#         data=CustomData(
#             gender=request.form.get('gender'),
#             race_ethnicity=request.form.get('race_ethnicity'),
#             parental_level_of_education=request.form.get('parental_level_of_education'),
#             lunch=request.form.get('lunch'),
#             test_preparation_course=request.form.get('test_preparation_course'),
#             reading_score=float(request.form.get('reading_score')),
#             writing_score=float(request.form.get('writing_score'))
#         )

#         pred_df=data.get_data_as_dataframe()
#         Predict_Pipeline=PredictPipeline()
#         results=Predict_Pipeline.predict(pred_df)

#         return render_template('index.html', results=results[0])  # FIXED
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and preprocessor
model = pickle.load(open("artifacts/model.pkl", "rb"))
preprocessor = pickle.load(open("artifacts/preprocessor.pkl", "rb"))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predictdata():
    if request.method == "GET":
        return render_template("home.html")
    
    try:
        # Get form data
        gender = request.form.get("gender")
        race_ethnicity = request.form.get("race_ethnicity")
        parental_level_of_education = request.form.get("parental_level_of_education")
        lunch = request.form.get("lunch")
        test_preparation_course = request.form.get("test_preparation_course")
        reading_score = float(request.form.get("reading_score"))
        writing_score = float(request.form.get("writing_score"))

        # Create dataframe with exact column names
        data = pd.DataFrame([{
            "gender": gender,
            "race_ethnicity": race_ethnicity,
            "parental_level_of_education": parental_level_of_education,
            "lunch": lunch,
            "test_preparation_course": test_preparation_course,
            "reading_score": reading_score,
            "writing_score": writing_score
        }])

        # Transform input data
        data_scaled = preprocessor.transform(data)

        # Predict
        pred = model.predict(data_scaled)[0]
        pred = float(pred)

        # Keep output within 0 to 100
        pred = max(0, min(100, pred))

        # Round result
        pred = round(pred, 2)

        return render_template("home.html", results=pred)

    except Exception as e:
        return render_template("home.html", results=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
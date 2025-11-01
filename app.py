from flask import Flask,request,render_template
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app = application

## Route for a homepage

@app.route('/')
def index():
    """Landing page route"""
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    """Route for prediction form and results"""
    if request.method=='GET':
        return render_template('home.html')
    else:
        try:
            # Get form data
            gender = request.form.get('gender')
            race_ethnicity = request.form.get('ethnicity')
            parental_level_of_education = request.form.get('parental_level_of_education')
            lunch = request.form.get('lunch')
            test_preparation_course = request.form.get('test_preparation_course')
            reading_score_str = request.form.get('reading_score')
            writing_score_str = request.form.get('writing_score')
            
            # Validate required fields
            if not all([gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course]):
                return render_template('home.html', error='All fields are required')
            
            # Validate and convert numeric fields
            if reading_score_str is None or reading_score_str == '':
                return render_template('home.html', error='Reading score is required')
            if writing_score_str is None or writing_score_str == '':
                return render_template('home.html', error='Writing score is required')
            
            try:
                reading_score = int(reading_score_str)
                writing_score = int(writing_score_str)
            except ValueError:
                return render_template('home.html', error='Reading and writing scores must be valid numbers')
            
            # Validate score ranges
            if not (0 <= reading_score <= 100):
                return render_template('home.html', error='Reading score must be between 0 and 100')
            if not (0 <= writing_score <= 100):
                return render_template('home.html', error='Writing score must be between 0 and 100')
            
            data=CustomData(
                gender=gender,
                race_ethnicity=race_ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                reading_score=reading_score,
                writing_score=writing_score,
            )
            pred_df=data.get_data_as_dataframe()

            predict_pipeline=PredictPipeline()
            results=predict_pipeline.predict(pred_df)
            return render_template('home.html',results=round(float(results[0]), 2))
            
        except ValueError as e:
            return render_template('home.html', error=f'Invalid input: {str(e)}')
        except Exception as e:
            return render_template('home.html', error=f'An error occurred: {str(e)}')

if __name__=="__main__":
    # Use environment variable for debug mode, default to False for security
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host="0.0.0.0", debug=debug_mode)
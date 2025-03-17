from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

class SentimentForm(FlaskForm):
    sentence = StringField("Enter a sentence", validators=[DataRequired()])
    submit = SubmitField("Analyze")

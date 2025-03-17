from flask import Flask, render_template
from forms import SentimentForm
from model import analyze_sentiment

app = Flask(__name__)
app.config["SECRET_KEY"] = "1234"  # Needed for WTForms security

@app.route("/", methods=["GET", "POST"])
def index():
    form = SentimentForm()
    message = ""
    
    if form.validate_on_submit():  # Checks if form is submitted
        user_input = form.sentence.data
        sentiment = analyze_sentiment(user_input)
        return render_template("result.html", message=sentiment)  # Redirect to result page
    
    return render_template("index.html", form=form)

if __name__ == "__main__":
    app.run(debug=True)

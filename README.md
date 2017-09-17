# Toy sentiment API

This is a toy API that will predict if a tweet is positive or negative. 

Download the toy API:

    git clone https://github.com/melvynator/toy_sentiment_API

Go into the main repository and create a virtual environement:

    cd toy_sentiment_API
    virtualenv -p python3 venv
    source venv/bin/activate
    Then install Flask and Scikit-Learn (For the machine learning)

pip install -r requirements.txt

Then you can launch your local server:

    python sentiment_server.py

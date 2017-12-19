import pickle
from flask import Flask, jsonify, render_template, redirect, url_for, request
from elasticsearch import Elasticsearch
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer
from nltk.corpus import movie_reviews
import re

app = Flask(__name__)

es = Elasticsearch([{"host": "localhost", "port": 9200}])


@app.route('/labeling', methods=['GET'])
def get_random_tweet():
    query = {
        "query": {
            "bool": {
                "must_not": {
                    "exists": {
                        "field": "sentiment"
                    }
                },
                "must": {
                    "match": {
                        "tweet_lang": "en"
                    }
                }
            }
        }
    }
    response = es.search(index="twitter", doc_type="tweet", body=query, size=1)
    tweet_id = response["hits"]["hits"][0]["_id"]
    return redirect(url_for("displaying_a_tweet", tweet_id=tweet_id))


@app.route('/labeling/<string:tweet_id>', methods=['GET'])
def displaying_a_tweet(tweet_id):
    print("The tweet id is:" + tweet_id)
    query = {
                "size": 1,
                "query": {
                    "match": {
                        "_id": tweet_id
                    }
                }
    }
    response = es.search(index="twitter", doc_type="tweet", body=query)
    return render_template("labeling.html", data=response)


@app.route('/labeling/<string:tweet_id>', methods=['POST'])
def label_a_tweet(tweet_id):
    sentiment = request.form['submit']
    query = {
                "size": 1,
                "query": {
                    "match": {
                        "_id": tweet_id
                    }
                }
    }
    response = es.search(index="twitter", doc_type="tweet", body=query)
    parent = response["hits"]["hits"][0]["_parent"]
    query = {
        "doc" : {
            "sentiment" : sentiment
        }
    }
    es.update(index='twitter',doc_type='tweet',id=tweet_id, parent=parent, body=query, refresh=True)
    return redirect(url_for("get_random_tweet"))


@app.route('/predict', methods=['POST'])
def predict_a_tweet():
    tweet_content = request.json["submit"]
    cleaned_tweet = clean_text(tweet_content)
    positive_likelyhood = text_blob(cleaned_tweet).sentiment.p_pos
    if 0.45 <= positive_likelyhood <= 0.55:
        return jsonify(sentiment="neutral", score=positive_likelyhood)
    if positive_likelyhood > 0.60:
        return jsonify(sentiment="positive", score=positive_likelyhood)
    else:
        return jsonify(sentiment="negative", score=positive_likelyhood)
    #vectorizer = pickle.load(open("models/vectorizer.p", 'rb'))
    #classifier = pickle.load(open("models/classifier.p", 'rb'))
    #tweet_content = request.json["submit"]
    #to_predict = [tweet_content]
    #tfidf_to_predict = vectorizer.transform(to_predict)
    #predicted = classifier.predict(tfidf_to_predict)
    #print(tweet_content, predicted[0])


def clean_text(tweet_to_clean):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet_to_clean).split())


if __name__ == '__main__':
    text_blob = Blobber(analyzer=NaiveBayesAnalyzer())
    app.run(host='0.0.0.0', debug=True)

#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 17:24:50 2018
@author: tittaya
"""
#!flask/bin/python

# from qas import *
import sys
import notebookutil as nbu
sys.meta_path.append(nbu.NotebookFinder())
import female as female
import male as male

from google.cloud import bigquery
import json
import os

from flask import Flask, jsonify, request, abort, render_template, redirect, url_for
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

project_id = 'robot-personnel'
client = bigquery.Client.from_service_account_json('robot-personnel-76469dafbe2b.json')


@app.route('/ask', methods=['POST'])
def ask():

    post_data = request.get_json()
    question = post_data.get('question')
    gender = post_data.get('gender')

    answer = model(question, gender)
    
    return  jsonify({'result': answer})

@app.route('/get_faqs', methods=['POST'])
def get_faqs():

    questions = []

    post_data = request.get_json()
    question = post_data.get('question')
    gender = post_data.get('gender')
    prefecture = post_data.get('prefecture')

    query = """ 
    SELECT question, 
    COUNT(*) FROM qa_data.free_answer 
    WHERE gender = @gender AND prefecture = @prefecture
    GROUP BY question 
    ORDER BY count(*) 
    DESC LIMIT 10; 
    """
    query_params = [
       bigquery.ScalarQueryParameter('gender', 'STRING', gender),
       bigquery.ScalarQueryParameter('prefecture', 'STRING', prefecture)
    ]

    job_config = bigquery.QueryJobConfig()
    job_config.query_parameters = query_params
    query_job = client.query(query, job_config=job_config)

    results = query_job.result()  # Waits for job to complete.

    for row in results:
        questions.append(row.question)
        # print("{} : {} views".format(row.question, row.f0_))

    return  jsonify({'result': questions})

@app.route('/evaluate_randomly', methods=['GET'])
def evaluate_randomly(model=None):
    question, answer, predicted, score = female.evaluate_randomly()

    data = {
        "question": question,
        "answer": answer,
        "predicted": predicted,
        "score": score,
    }

    json_data = json.dumps(data, ensure_ascii=False)
    
    return jsonify(json_data)

def model(question, gender):

    if gender == '女性':
        answer = female.ask(question)
    else:
        answer = male.ask(question)

    return answer

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
    
    
    
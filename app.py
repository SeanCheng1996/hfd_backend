from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

import json
import pandas as pd

import sys
sys.path.append('../utils')
from utils.preprocessing import InitPreprocess
from helper import CA


prep = InitPreprocess()
main_df_path = 'HFD MVA History_modified.xlsx'
main_df = pd.read_excel(main_df_path)
main_df1 = prep.cleaning_pipeline(main_df)


app = Flask(__name__)
CORS(app, support_credentials=True)

@app.route("/")
@cross_origin(supports_credentials=True)
def hello():
    return "hello"


@app.route("/get_ca_data", methods=['POST'])
@cross_origin(supports_credentials=True)
def get_ca_data():
    ## mode1 and mode2 can take values {'principal', 'standard'}
    mode1 = 'principal'
    mode2 = 'principal'
    if 'row' in request.form:
        row=request.form['row']
    if 'col' in request.form:
        col=request.form['col']
    if 'mode1' in request.form:
        mode1 = request.form['mode1']
    if 'mode2' in request.form:
        mode2 = request.form['mode2']

    perform_CA = CA(data=main_df1, active_row=row, active_col=col)
    ca_data = perform_CA.generate_ca_2d(row_mode=mode1, col_mode=mode2)
    return ca_data

@app.route("/get_all_variables", methods=['POST'])
@cross_origin(supports_credentials=True)
def get_all_variables():
    # return {"all_variables":['Type of Collosion', 'General Location of Collision']}
    variables = []
    for col in main_df1.columns:
        ## extract all those variables having > 2 distinct categories, otherwise the variable
        ## cannot be active variable (row or column)
        if len(main_df1[col].unique()) > 2 and col not in ['Payroll', 'Shop Number', 'Station']:
            variables.append(col)
    return {'all_variables': variables}


@app.route("/get_stats", methods=['POST'])
@cross_origin(supports_credentials=True)
def get_stats():
    # variable1="Type of Collision"
    # variable2="General Location of Collision"
    if 'row' in request.form:
        row=request.form['row']
    if 'col' in request.form:
        col=request.form['col']

    perform_CA = CA(data=main_df1, active_row=row, active_col=col)
    return perform_CA.generate_stats()


# pie chart
@app.route("/get_profile", methods=['POST'])
@cross_origin(supports_credentials=True)
def get_profile():
    # direction: ['row', 'col']
    # category: ['mean', categories of the direction]
    if 'row' in request.form:
        row=request.form['row']
    if 'col' in request.form:
        col=request.form['col']
    if 'direction' in request.form:
        direction = request.form['direction']
    if 'category' in request.form:
        category = request.form['category']

    perform_CA = CA(data=main_df1, active_row=row, active_col=col)
    return perform_CA.generate_profile(direction=direction, category=category)

@app.route("/get_categories_for_pie", methods=['POST'])
@cross_origin(supports_credentials=True)
def get_categories_for_pie():
    # return [categories] for `row` or `col`
    if 'row' in request.form:
        row=request.form['row']
    if 'col' in request.form:
        col=request.form['col']
    if 'direction' in request.form:
        direction = request.form['direction']
    
    perform_CA = CA(data=main_df1, active_row=row, active_col=col)
    if direction == 'row':
        return {'pie_categories': list(perform_CA.count.index)}
    return {'pie_categories': list(perform_CA.count.columns)}

# stats
@app.route("/get_principal_inertia", methods=['POST'])
@cross_origin(supports_credentials=True)
def get_principal_inertia():
    '''
    Returns a table of inertia (think variation) by dimension (dim0 is 
    the x-axis, dim1 is the y-axis).
    '''
    return_dim=2
    if 'row' in request.form:
        row=request.form['row']
    if 'col' in request.form:
        col=request.form['col']
    if 'return_dim' in request.form:
        return_dim = request.form['return_dim']
        return_dim=int(return_dim)
    print(row)
    print(col)
    print(return_dim)

    perform_CA = CA(data=main_df1, active_row=row, active_col=col)
    return perform_CA.generate_principal_inertia(return_dim=return_dim)


@app.route("/get_count", methods=['POST'])
@cross_origin(supports_credentials=True)
def get_count():
    '''
    Returns the contigency table (frequency count by row and column categories)
    '''
    if 'row' in request.form:
        row=request.form['row']
    if 'col' in request.form:
        col=request.form['col']
    perform_CA = CA(data=main_df1, active_row=row, active_col=col)
    return perform_CA.count.to_json()


@app.route("/get_prob", methods=['POST'])
@cross_origin(supports_credentials=True)
def get_prob():
    '''
    Returns the expected probability table [p_ij] * total_num_collisions (
    i.e. expected frequency count by row and column categories)
    '''
    if 'row' in request.form:
        row=request.form['row']
    if 'col' in request.form:
        col=request.form['col']
    perform_CA = CA(data=main_df1, active_row=row, active_col=col)
    prob_table = perform_CA.expected * perform_CA.count.sum().sum()
    return prob_table.to_json()


@app.route("/get_chi2", methods=['POST'])
@cross_origin(supports_credentials=True)
def get_chi2(scale=100):
    '''
    Returns the chi-square value table same shape as the contigency table
    '''
    if 'row' in request.form:
        row=request.form['row']
    if 'col' in request.form:
        col=request.form['col']
    perform_CA = CA(data=main_df1, active_row=row, active_col=col)
    chi2_table = perform_CA.chi2 * scale
    return chi2_table.to_json()


if __name__ == '__main__':
    #app.run(host='127.0.0.1', port=8000, debug=True)
    app.run()
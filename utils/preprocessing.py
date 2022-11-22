# -*- coding: utf-8 -*-
"""MVA_Preprocessing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1o67KRNY3HQ4uT3f8uzBCsZM8qCAYl9Dg
"""

import os
from datetime import datetime
import re

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import numpy as np

from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

class InitPreprocess:
    ## columns already cleaned
    cleaned_cols = ['Payroll',
                    'Citation',
                    'Cause Due to Blocking',
                    'Emergency Run',
                    'Spotter Used']

    ## columns to be dropped
    drop_cols = ['Debit Day',
                'Vehicle Type',
                'Number of Spotters and Locations',
                'Item Type',
                'Active Incident']
    for num in [2, 3, 4, 5, 7, 8, 9]:
        drop_cols.append('Column' + str(num))

    ## columns to be basic-processed
    basic_process_cols = ['General Location of Accident', 
                            'Location of Damage', 
                            'Type of Accident', 
                            'Object Struck', 
                            'Shop Number',
                            'Rank', 
                            'Unit Type',
                            'Film']

    ## columns to be deep-processed
    deep_process_rulings = ['Ruling-Accident Review Board',
                        'Status-Final Staff Services Ruling']
    ## ['Station', 'Work Shift'] also needs to be deep-processed, but they have their own functions

    ## use 'Date of Accident' to generate 'Time of Day' variable according to rules specified in `time_dict`
    time_dict = {'Night': (0, 6), 'Morning': (6, 12), 'Afternoon': (12, 18), 'Evening': (18, 24)}

    ## debatable columns (drop for now)
    debatable_cols = ['Notes', 
                        'Location of Accident',
                        'Apparatus Number', 
                        'Column6']

    ## values contained in `basic_process_cols` to be processed as Unknown values
    unknowns_basic = ['[Please select]', '[Please Select]', 'Unknown']

    ## values contained in `deep_process_cols` to be processed as Unknown values
    unknowns_rulings = ['[Please select]', 'Unknown', 'Undetermined', 'For Information Only', 'Need More Information']

    ## groups of unique values in `Station` column to be unified; excluded valid station numbers
    station_dict = {
        'hfd_adminBuilding': ['Staff', 'Jefferson', 'ICO', 'PSO', 'EAP', 'Headquarters', 'PIO', 'Admin', 'Affairs', 'Chief', 'Resouce Mgt.', 'Emergency', '1801 Smith', 'MD', 'Logistics', 'Recruiting', 'SS', 'HR', 'HQ', 'Family Assistance'],
        'emergency_comm' : ['Communications'],
        'hfd_arson' : ['Arson', '1205', '1201', 'Dart', 'Shop', 'Repair', 'Supply', 'FMD', 'Fleet'],
        'hfd_saftyBureau' : ['BRAC', 'Insp', 'Fire Prevention', 'Fire Marshall'],
        '5': ['D-5'],
        'Other': ['FP', 'FP ', 'F/P', 'Sm. Engine', 'Small Engine', 'Airport', 'Air', 'EMS HQ', 'Ems HQ', 'VJTF', 'Res Comm', 'OEC', 'Small Engine', 'Rescue', 'ARFF', 'Support', 'Civilian', 'Mech', 'Sr Im Clerk', '0698', '1050', '301', '0690', '1050']
    }


    ## change deep to deep_copy
    def changing_values(self, data, columns, val_x, val_y, fill_nan=False, deep=True):
        """
        This is a helper function which changes all the entries with [val_x] to [val_y] in [columns] of [data]
        If [fill_nan]: imputing all nan values in the columns with [val_y]
        If [deep]: making a deep copy instead of modifying the original input [data]
        Input:
            val_x = a list of values to be changed
            val_y = a string of target value
        Output:
            [data1] = a deep copy of [data] with column in [columns] processed
        """
        data1 = data.copy(deep=deep)
        for col in columns:
            data1[col] = data1[col].apply(lambda x: val_y if x in val_x else x)
            if fill_nan:
                data1[col] = data1[col].fillna(value=val_y)
        return data1


    def basic_process(self, data, columns, unknowns, missing_val_sub='Other', fill_nan=True, deep=True):
        """
        This function process the `basic_process_cols` columns.
        Input:
            data: pandas DataFrame that containing [columns]
            columns: an array of column names, organized in `baisc_process_cols`
            unknowns: an array of unknown values to be replaced, e.g. ['[Please select]', 'Unknown'], organized in `unknowns_basic`
            missing_val_sub: a string to replace missing/undetermined values
            fill_nan: a boolean indicating whether to fill in the NaN's; if True, will replace NaN's with [missing_val_sub]
            deep: a boolean indicating whether to return a copy of [data]
        Output:
            [data1] = a copy of the dataframe [data] with column in [columns] processed
        """
        data1 = self.changing_values(data, columns, val_x=unknowns, val_y=missing_val_sub, fill_nan=fill_nan, deep=deep)
        data1 = self.changing_values(data1, columns=['Unit Type'], val_x=['DC'], val_y='District Chief')
        data1 = self.changing_values(data1, columns=['Unit Type'], val_x=['Basic'], val_y='Ambulance')
        data1 = self.changing_values(data1, columns=['Rank'], val_x=['Enginner/Operator'], val_y='Engineer/Operator')
        data1 = self.changing_values(data1, columns=['Film'], val_x=[1.0], val_y='True')
        data1 = self.changing_values(data1, columns=['Film'], val_x=[0.0], val_y='False')
        return data1


    def processing_rulings(self, data, columns, unknowns, missing_val_sub='Other', fill_nan=True, deep=True):
        '''
        This function process the `deep_process_rulings` column groups into a binary column (with 'Other' 
        replacing all undetermined/missing entries).

        Input:
            data: pandas DataFrame that containing [columns]
            columns: an array of column names, organized in `deep_process_rulings`
            unknowns: an array of unknown values to be replaced, e.g. ['[Please select]', 'Unknown'], organized in `unknowns_rulings`
            missing_val_sub: a string to replace missing/undetermined values
            fill_nan: a boolean indicating whether to fill in the NaN's; if True, will replace NaN's with [missing_val_sub]
            deep: a boolean indicating whether to return a copy of [data]
        Output:
            [data1] = a copy of the dataframe [data] with column in [columns] processed
        '''
        ## first process: fill in and unify missing values (collected in the array `unknowns`)
        data1 = self.changing_values(data, columns, val_x=unknowns, val_y=missing_val_sub, fill_nan=fill_nan, deep=deep)
        ## second process: for all entries containing 'incident' or 'other', change to 'other'
        ## else if containing 'non', change to 0; 
        ## else change to 1 (preventable).
        for col in columns:
            data1[col] = data1[col].apply(lambda x: 'Other' if 'incident' in x.lower() or 'other' in x.lower() else 'Non-preventable' if 'non' in x.lower() else 'Preventable')
        ## merge these two ruling columns based on their relation: 0 and 'other' in final ruling follow the result in review board. 1 in final ruing remains.
        data1['Merged Ruling']=data1[columns[1]]
        for index in range(len(data1[columns[0]])):
            if data1[columns[1]][index]==1:
                continue
            data1['Merged Ruling'][index]=data1[columns[0]][index]
        return data1


    def process_station(self, data, station_dict, missing_val_sub='Other', deep=True):
        '''
        This function process the `Station` column.
        Input:
            data: pandas DataFrame that contains the `Station` column
            station_dict: a dictionary used to categorize the unique values in `Station` into groups based on
                        keywords contained the unique values, e.g., all unique values in `Station` containing 
                        'Staff' belong to the HFD Administrative Building, so `Staff` is a member of the
                        VALUE array associated with KEY='hfd_adminBuilding`
            missing_val_sub: a string to replace missing/undetermined values
        Output:
            [data1] = a copy of the dataframe [data] with `Station` column processed
        '''
        data1 = data.copy(deep=deep)
        def check_station_group(x, col_dict, missing_val_sub=missing_val_sub):
            if str(x) != 'nan':
                return_ref = x
                for key in col_dict:
                    for item in col_dict[key]:
                        if x.lower() in item.lower() or item.lower() in x.lower():
                            return_ref = key
                return return_ref
            else:
                return missing_val_sub
        def format_station_num(x):
            if str(x) != 'nan':
                if bool(re.search(r'\d', x)):
                    ## NOTE: assuming if it contains integer, the whole string are convertible to INT
                    return int(x)
                else:
                    return x
            else:
                return missing_val_sub
        data1['Station'] = data1['Station'].apply(lambda x: format_station_num(check_station_group(x, station_dict)))
        return data1


    def process_workshift(self, data, missing_val_sub='Other', deep=True):
        '''
        This function process the `Work Shift` column: 
                1. change all unclear/missing values to [missing_val_sub];
                2. for all other values, take the first letter in a/b/c/d as the workshift assignment.
        Input:
            data: pandas DataFrame that contains the `Work Shift` column
            missing_val_sub: a string to replace missing/undetermined values
        Output:
            [data1] = a copy of the dataframe [data] with `Work Shift` column processed
        '''
        data1 = data.copy(deep=deep)
        def check_workshift_group(x, missing_val_sub=missing_val_sub):
            if str(x) != 'nan':
                if x.lower() in ['a', 'b', 'c', 'd']:
                    return x.upper()
                elif bool(re.match('^[abcd][/-]', x.lower())):
                    return x[0].upper()
                else:
                    return missing_val_sub
            else:
                return missing_val_sub
        data1['Work Shift'] = data1['Work Shift'].apply(lambda x: check_workshift_group(x))
        return data1
    

    def process_hour(self, data, time_dict, deep=True):
        '''
        This function extracts hours from 'Date of Accident' variable and characterize a colliison into 
        ['Night', 'Morning', 'Afternoon', 'Evening].
        Input:
            data: pandas DataFrame that contains the `Date of Accident` column
            time_dict: python dictionary specifying rules for dividing hours into discrete categories
        
        Output:
            data: dataframe with 'Time of Day' variable added
        '''
        def generate_time_of_day(hour, time_dict):
            for key in time_dict:
                (lo, hi) = time_dict[key]
                if hour >= lo and hour < hi:
                    return key
        data1 = data.copy(deep=deep)
        data1['Time of Day'] = data1['Date of Accident'].apply(lambda x: generate_time_of_day(x.hour, time_dict) if generate_time_of_day(x.hour, time_dict) else 'Missing')
        return data1


    def modify_by_freq(self, data, column_name='Payroll', clip_threshold=10, clip_flag=True):
        '''
        This function aims to cluster payroll numbers according to their frequency, so that decrease the number of unique values.
        A new column will be added, named 'freq_modified_'+column_name
        Input:
            data: pandas DataFrame that contains the `Payroll` column
            column_name： default as 'Payroll'. But can be changed if any other columns need to be replaced by their counts.
            threshold: only affect when clip_flag is True. Any frequency above threshold will be regarded as threshold
            clip_flag: True to clip frequency, False to turn off clipping.
        
        Output:
            data: the modified dataFrame
        '''
        data['Modified by Freq-'+column_name]=data[column_name]
        curData=data['Modified by Freq-'+column_name]
        statsDict=curData.value_counts().to_dict()
        for index in range(len(curData)):
            if curData[index] in statsDict:
                # curData[index]=float(statsDict[curData[index]]) if not clip_flag else clip_threshold if statsDict[curData[index]]>=clip_threshold else float(statsDict[curData[index]])
                curData[index] = 'Count ' + str(int(statsDict[curData[index]])) if not clip_flag else 'Count >=' + str(int(clip_threshold)) if statsDict[curData[index]]>=clip_threshold else 'Count ' + str(int(statsDict[curData[index]]))
        return data


    def modify_by_freq_acc(self, data, column_name='Payroll', missing_val_sub=0):
        '''
        This function aims to cluster payroll numbers according to their frequency, so that decrease the number of unique values.
        A new column will be added, named 'freq_modified_'+column_name
        Input:
            data: pandas DataFrame that contains the `Payroll` column
            column_name： default as 'Payroll'. But can be changed if any other columns need to be replaced by their counts.
            threshold: only affect when clip_flag is True. Any frequency above threshold will be regarded as threshold
            clip_flag: True to clip frequency, False to turn off clipping.
        
        Output:
            data: the modified dataFrame
        '''
        df = data.copy()
        ordered_df = df.sort_values(by = 'Date of Accident')[[column_name, 'Date of Accident']]
        freq_acc = {}
        for num in ordered_df[column_name].unique():
            freq_desc = ordered_df[ordered_df[column_name] == num]
            for i, id in enumerate(freq_desc.index.values):
                freq_acc[id] = i
        df['id'] = df.index.values
        df['Modified by FreqAcc-'+column_name] = df['id'].apply(lambda x: freq_acc.get(x))
        df['Modified by FreqAcc-'+column_name] = df['Modified by FreqAcc-'+column_name].fillna(missing_val_sub)
        df = df.drop(columns=['id'])
        return df


    def cleaning_pipeline(self, data, 
                            drop_cols=drop_cols, debatable_cols=debatable_cols, time_dict=time_dict,
                            basic_process_cols=basic_process_cols, deep_process_rulings=deep_process_rulings, 
                            unknowns_basic=unknowns_basic, unknowns_rulings=unknowns_rulings, station_dict=station_dict,
                            drop_debatable=True, do_process_station=True, do_process_workshift=True, freq_acc=False, rename_acc=True):
        '''
        This is the pipeline function that process the entire dataframe [data].
        NOTE: to NOT drop debatable columns (e.g. Notes, Location of Accident), set `drop_debatable=False`.
        
        Input:
            data: pandas dataframe
            drop_cols: a list of column names to be dropped
            debatable_cols: a list of column names that are undecided what to do (set [drop_debatable] = True to drop)
            basic_process_cols: a list of column names that requires only baisc processing -- i.e. unifying missing values
            deep_process_rulings: a list of PREVENTABILITY columns (two columns) that requires special processing 
            unknowns_basic: a list of unknown value representations in [basic_process_cols]
            unknowns_rulings: a list of unknown value representations in [deep_process_rulings]
            station_dict: a dictionary indicating the grouping of unique values in `Station` column
            basic_process: function used to process [basic_process_cols]
            processing_rulings: function used to process [deep_process_rulings]
            process_station: function used to process `Station`
            process_workshift: function used to process `Work Shift`
            drop_debatable: boolean indicating whether to drop the list of debatable columns
            do_process_station: boolean indicating whether to process station
            do_process_workship: boolean to indicate whether to process workshift
            freq_acc: boolean; if True, use accumulative frequency to process, else use overall frequency
            rename_acc: boolean; if True, renames the appearance of 'Accident' to 'Collision'
        Output:
            [data1]: a copy of dataframe [data] with columns processed as required
        '''
        ## add modify by accumulative frequency. columns: Payroll, Shop Number
        data1 = data.copy()
        if freq_acc:
            data1 = self.modify_by_freq_acc(data)
            data1 = self.modify_by_freq_acc(data1, column_name='Shop Number')

        ## dropping columns
        if drop_debatable:
            drop_cols = drop_cols + debatable_cols
        data1 = data1.drop(columns=drop_cols)

        ## basic processing columns
        ## unknowns_basic = ['[Please select]', 'Unknown']
        data1 = self.basic_process(data1, columns=basic_process_cols, unknowns=unknowns_basic)

        ## deep processing columns (GROUP 1 -- preventability rulings)
        ## unknowns_rulings = ['[Please select]', 'Unknown', 'Undetermined', 'For Information Only', 'Need More Information']
        data1 = self.processing_rulings(data1, columns=deep_process_rulings, unknowns=unknowns_rulings)

        ## generate 'Time of Day' variable
        data1 = self.process_hour(data1, time_dict=time_dict)

        ## modify value by frequecy. columns: Payroll, Shop Number
        ## keep modify freq by total or acc
        if not freq_acc:
            data1 = self.modify_by_freq(data1)
            data1 = self.modify_by_freq(data1, column_name='Shop Number')

        ## deep processing columns (Station)
        if do_process_station:
            data1 = self.process_station(data1, station_dict=station_dict)
        
        ## deep processing columns (Workshift)
        if do_process_workshift:
            data1 = self.process_workshift(data1)
        
        ## rename 'Accident' to 'Collision' in column names
        if rename_acc:
            data1.rename(lambda col_name: col_name.replace('Accident', 'Collision'), axis=1, inplace=True)
        
        return data1


    def encode_to_num(self, df, column_name, do_replace=False):
        '''
        This function encode and replace the data with a number, do label encoding.
        It can be used as just encoding, rather than replacing.
        Input:
            df: dataFrame to be processed
            column_name: column to be processed, should be a single str
            do_replace: If true, replace oriValue with encoded value; Else, remain original values.
        Output:
            [oriValues]: original values in this column, a np.ndarray
            [encodedValues]: encoded values which are corresponding to oriValues, a np.ndarray
            [mapping]: a dict, record the relationship between ori and encoded value --- {key:encoded, value: ori}
        '''

        label_encoder = preprocessing.LabelEncoder()

        # Encode labels in column
        oriValues = df[column_name].unique()
        encodedValues = label_encoder.fit_transform(df[column_name].unique())

        if do_replace:
            df[column_name] = label_encoder.fit_transform(df[column_name])

        mapping = {}
        for i in range(len(oriValues)):
            mapping[encodedValues[i]] = oriValues[i]

        return oriValues, encodedValues, mapping

    def encode_to_num_multiple(self, df, columns, do_replace=False):
        '''
        Encode multiple columns, replace the data with number,
        Input:
            df: dataFrame to be processed
            columns: columns to be processed, should be a str list
            do_replace: If true, replace oriValue with encoded value; Else, remain original values.
        Output:
            [mapping]: a dict, record the relationship between ori and encoded value \
                {
                    column: {key:encoded, value: ori}
                    ....
                }
        '''
        mappings = {}
        for column in columns:
            print(f'encoding {column}...')
            _, _, mapping = self.encode_to_num(df, column, do_replace)
            mappings[column] = mapping
        return mappings

    def hello(self):
        print("hello from Preprocessing.FirstPreprocess")
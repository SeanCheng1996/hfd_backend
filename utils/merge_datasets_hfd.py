
import pandas as pd
import datetime
import re 
import numpy as np
from datetime import datetime

"""
Merge MVA History data with HFD Station information based on Station number. 
Station number can be found in the 'Station' column of both these datasets. 

Interestingly, station numbers are also included as part of the apparatus numbers. 
However, the station numbers included this way appear to sometimes be off from the actual station. 
This inconsistency is likely due to instances where a reserve apparatus was involved in an incident. 
The number part of the apparatus number should reference the station that the vehicle was assigned to on the date of the accident. 
But, in some instances, it references a randomly assigned number by HFD's Fleet Maintenance Division.
"""

# import main mva data as mva_df
mva_df_path = '../data/HFD MVA History_modified.xlsx'
mva_df = pd.read_excel(mva_df_path)
mva_df.info()

# import HFD station data as station_df
station_df_path = '../data/HFD Stations.xlsx'
station_df = pd.read_excel(station_df_path)
station_df.info()

station_df.columns=station_df.iloc[0]
# drop unused first col & rows at end that do not correspond to column names
station_df = station_df.iloc[1:-21 , 1:]

# match the types of the Station columns
mva_df['Station'] = mva_df['Station'].astype(str)
station_df['Station'] = station_df['Station'].astype(str)

# remove leading zeros 
mva_df['Station'] = ([num.lstrip("0") for num in mva_df['Station']])

mva_station = pd.merge(mva_df,station_df, on='Station', how='left')

"""
Merge this dataset with MVA Costs 17-20 data. 
Merge based on vehicle 'Shop Number' and 'Accident Date' which can be found in both the datasets. 
"""
# import mva costs 2017-2020 data
costs_df_path = '../data/MVA Costs - 17-20.xlsx'
# data is given in separate excel sheets
costs_sheets = pd.read_excel(costs_df_path, sheet_name=None)
# combine the sheets
costs_df = pd.concat(costs_sheets[frame] for frame in costs_sheets.keys())
costs_df.info()

# merge main dataset with costs data based on Shop Number and Date of Accident
# create Accident Date in both dataset - both with same format: YYYY-MM-DD
mva_station['Accident Date'] = pd.to_datetime(mva_station['Date of Accident'], 
                                        errors='coerce').dt.strftime('%Y-%m-%d')
costs_df['Accident Date'] = pd.to_datetime(costs_df['Date of Accident'],
                                        errors='coerce').dt.strftime('%Y-%m-%d')
# relabel 
costs_df['Shop Number'] = costs_df['Shop #']

mva_costs_station = pd.merge(mva_station, costs_df, 
                             on=['Shop Number', 'Accident Date'], how='left')

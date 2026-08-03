import pandas as pd
import numpy as np
import warnings




stores_info = pd.read_csv('store.csv')
sales_info = pd.read_csv('train.csv')



stores_drop_cols = ['ContinuousBogoMonths', 'ContinuousBogoSinceWeek', 
                    'ContinuousBogoSinceYear', 'RivalOpeningMonth', 
                    'RivalEntryYear', 'RetailType', 'Stock variety']



sales_drop_cols = ['Holiday', 'NumberOfCustomers']

    
stores_info = stores_info.drop(columns=[c for c in stores_drop_cols if c in stores_info.columns])
sales_info = sales_info.drop(columns=[c for c in sales_drop_cols if c in sales_info.columns])

    
merged_data = pd.merge(stores_info, sales_info, on='Store_id')

    
merged_data['Date'] = pd.to_datetime(merged_data['Date'])
    

merged_data = merged_data.sort_values(by='Date')

    
merged_data['Year'] = merged_data['Date'].dt.year
merged_data['Month'] = merged_data['Date'].dt.month
merged_data['Day'] = merged_data['Date'].dt.day
merged_data['DayOfYear'] = merged_data['Date'].dt.day_of_year 
merged_data['WeekOfYear'] = (merged_data['DayOfYear'] - 1) // 7 + 1
merged_data = merged_data.drop(columns=['DayOfYear'])

   
# Replace missing values in 'DistanceToRivalStore' with the median
if 'DistanceToRivalStore' in merged_data.columns:
        median_distance = merged_data['DistanceToRivalStore'].median()
        merged_data['DistanceToRivalStore'] = merged_data['DistanceToRivalStore'].fillna(median_distance)

# Replace the rest of missing values with 0
merged_data = merged_data.fillna(0)

    

merged_data = merged_data.sort_values('Date')
merged_data = merged_data.drop(columns=['Date', 'Store_id'])


output_filename = 'processed_data.csv'
merged_data.to_csv(output_filename, index=False)
    


# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# Cut out bottom and top 1% of price data
def price_percent_cut(df_NEW, col):
    print("    Cutting out 1% of the data")
    #find the top 1%
    price_low = np.percentile(df_NEW[col].values, 1)
    price_high = np.percentile(df_NEW[col].values, 99)
    
    
    #cut out the defined range above from the dataframe
    df_NEW = df_NEW.drop(df_NEW[df_NEW.col < price_low].index)
    df_NEW = df_NEW.drop(df_NEW[df_NEW.col > price_high].index)

    return df_NEW

# Create all extra simple features
def newFeatures(initial_df):
    # convert created column into datetime type
    print("    Creating new features")
    try:
        # Create datetime features
        initial_df['DateTime'] = pd.to_datetime(initial_df.created)
        initial_df['day_created'] = initial_df.DateTime.map(lambda x: x.day)
        initial_df['month_created'] = initial_df.DateTime.map(lambda x: x.month)
        initial_df['year_created'] = initial_df.DateTime.map(lambda x: x.year)
        initial_df['hour_created'] = initial_df.DateTime.map(lambda x: x.hour)
        initial_df['day_of_week_created'] = initial_df.DateTime.map(lambda x: x.dayofweek)
        initial_df.drop('created', axis=1, inplace=True)

        # create feature for number of photos, features and description length
        initial_df['num_of_photos'] = initial_df.photos.map(len)
        initial_df['num_of_features'] = initial_df.features.map(len)
        initial_df['description_length'] = initial_df.description.apply(lambda x: len(x.split(" ")))
        initial_df['studio'] = initial_df.bedrooms.apply(lambda x: 1 if x==0 else 0)
        
        # Log price and square feet
        initial_df['log_price'] = initial_df.price.map(np.log)
        initial_df['price_sq'] = initial_df.price.map(np.square)
        
        # price per bedroom
        initial_df.bedrooms[initial_df.bedrooms == 0] = 1
        initial_df['price_per_bedroom'] = initial_df.price / initial_df.bedrooms
    except:
        print("    Clean_Preprocessed function skipped as it can only be run once")
    return initial_df

# Remove prices outside of defined range
def price_outliers(df_NEW, price_low, price_high):
    print("    Removing prices outside of range:",price_low,"-",price_high,"...")
    df_NEW = df_NEW.drop(df_NEW[df_NEW.price < price_low].index)
    df_NEW = df_NEW.drop(df_NEW[df_NEW.price > price_high].index)
    
    return df_NEW

# Remove locations outside of New York
def remove_nonNY_coords(df_NEW, ny_boundaries):
    print("    Removing rentals outside of NY...")
    #Removing out of bounds longitude
    df_NEW = df_NEW.drop(df_NEW[df_NEW.longitude < ny_boundaries[0]].index)
    df_NEW = df_NEW.drop(df_NEW[df_NEW.longitude > ny_boundaries[1]].index)

    #Removing out of bounds latitude
    df_NEW = df_NEW.drop(df_NEW[df_NEW.latitude < ny_boundaries[2]].index)
    df_NEW = df_NEW.drop(df_NEW[df_NEW.latitude > ny_boundaries[3]].index)

    return df_NEW

# FInd the euclidian distance
def euclid_dist(x, lat, long):
    return np.sqrt((x[0]-lat)**2 + (x[1]-long)**2)

# Create euclidian distance to each bourogh based on longitude and latitude
def boroughs(df):
    print("    Creating boroughs feature...")
# distance from borough centres
    the_bronx     = [40.8448, -73.8648]
    manhattan     = [40.7831, -73.9712]
    queens        = [40.7282, -73.7949]
    brooklyn      = [40.6782, -73.9442]
    staten_island = [40.5795, -74.1502]
    
    # List of the boroughs
    borough_list = {'the_bronx': the_bronx,
                    'manhattan': manhattan,
                    'queens': queens,
                    'brooklyn': brooklyn,
                    'staten_island': staten_island}
    
    for key in borough_list:
        df[key] = df[['latitude','longitude']].apply(euclid_dist, 
                                                     args=(borough_list[key]), 
                                                     axis=1)
    return df

# Assess each ID_quality 
# 1 get each ID
# 2 get number of listings that are high or medium
# 3 percentage of listings 
def makeFeatureQuality(strName,df):
    print("    Creating",strName,"quality feature")
    QualityTemp = (df.groupby(strName)['interest_level'].apply(list)).to_dict()
    
    for key in QualityTemp:
        qualList = QualityTemp[key]
        listLength = len(qualList)
        totalScore = 1
        for item in qualList:
            if item == "low":
                item = 0
            elif item == "medium":
                item = 1
            elif item == "high":
                item = 1
            else:
                item = -99999
            totalScore =+ item
        totalScore = totalScore / listLength
        QualityTemp[key] = [totalScore]
    return QualityTemp

# Main function that runs all the above pre
def main(df,train):
#==============================================================================
# Control panel for price and location data
#==============================================================================
    #price_low = 1000    # To set cutoff manually instead of by %
    #price_high = 10000
    price_low = np.percentile(df['price'].values, 1)
    price_high = np.percentile(df['price'].values, 99)
    
    # Define upper and lower limits for NewYork
    long_low  = -74.1
    long_high = -73.6
    lat_low   =  35
    lat_high  =  41
    ny_boundaries = [long_low,long_high,lat_low,lat_high]
#==============================================================================
# Clean data and show how many rows of data are removed at each step
#==============================================================================
    dataCount = len(df)

    df = newFeatures(df)
    print("")
    print("Running set:", train)
    print("Datapoints:",dataCount)
    print("Features",len(df.columns))
    newCount = len(df)
    dataCount=newCount
    
    if train:
        df = remove_nonNY_coords(df, ny_boundaries)
    newCount= len(df)
    print(dataCount-newCount,"rentals outside NY removed")
    dataCount=newCount

    if train:
        df = price_outliers(df, price_low, price_high)
    newCount= len(df)
    print(dataCount-newCount,"datapoints outside price range")

    df = boroughs(df)
    
#==============================================================================
# Building occurances and listings for each broker
#==============================================================================
    def getOccurances(strName):
        QualityTemp = (df.groupby(strName)['interest_level'].apply(list)).to_dict()
        
        for key in QualityTemp:
            qualList = QualityTemp[key]
            listLength = len(qualList)
            totalScore = 1
            for item in qualList:
                if item == "low":
                    item = 0
                elif item == "medium":
                    item = 1
                elif item == "high":
                    item = 1
                else:
                    item = -99999
                totalScore =+ item
            totalScore = totalScore / listLength
            QualityTemp[key] = listLength
        return QualityTemp
    
    # adding the new value to the dataframe
    managerID = 'manager_id'
    buildingID = 'building_id'
    mangagerQuality = getOccurances(managerID)
    buildingQuality = getOccurances(buildingID)
    
    df["mangager_num_listings"] = ""  
    df["building_num_occurances"] = "" 
    df["mangager_num_listings"] = df[managerID].map(mangagerQuality)
    df["building_num_occurances"] = df[buildingID].map(buildingQuality)
        
    
    # adding building and manager quality 
    managerQuality = 0
    buildingQuality = 0
    
    if train:
        managerID = 'manager_id'
        buildingID = 'building_id'
        managerQuality = makeFeatureQuality(managerID,df)
        buildingQuality = makeFeatureQuality(buildingID,df)
        
#        df["manager_quality"] = ""  
#        df["building_quality"] = ""   
        df["manager_quality"] = df[managerID].map(managerQuality)
        df["manager_quality"] = df.manager_quality.apply(lambda x: x[0])
        df["building_quality"] = df[buildingID].map(buildingQuality)
        df["building_quality"] = df.building_quality.apply(lambda x: x[0])    

    print("*********************")
    print("Preprocessing DONE...")
    print("Datapoints:",len(df))
    print("Features",len(df.columns))
    
    if train:
        return df, managerQuality, buildingQuality
    else:
        return df
    
    

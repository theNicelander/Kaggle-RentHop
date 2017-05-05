# Kaggle-RentHop
Final project for Data Analytics class, QMUL 2017. Dataset and challenge taken from Kaggle:

https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries

Using Supervised machine learning, find what makes a particular apartment rental stand out and get high interest. 
Project written in Python, in Spyder IDE. Main third-party libraries which were used are 
- SciKit learn
- Pandas 
- MatplotLib.

Below is a quick summary of the main points of the analysis - Full report is available in pdf (31 pages)

![cloud-petur-einarsson](https://github.com/Hunang/Kaggle-RentHop/blob/master/Images/Word-Cloud.PNG "cloud-petur-einarsson")

## Workflow
The first step was to understand RentHop, read the declaration from the spokesperson and explore the assignment given to get a sense of direction as to where to start. Thereafter, the following workflow was implemented until the end of the project:

![workflow-petur-einarsson](https://github.com/Hunang/Kaggle-RentHop/blob/master/Images/workflow.PNG "workflow-petur-einarsson")

## Data - EDA and cleaning
Initial features 

![features-petur-einarsson](https://github.com/Hunang/Kaggle-RentHop/blob/master/Images/features.PNG "features-petur-einarsson")

The vast majority, or 69% of the data, contains low-interest rentals and only 8% is high-interest apartments. This means the dataset is very imbalanced.

![imbalance-petur-einarsson](https://github.com/Hunang/Kaggle-RentHop/blob/master/Images/interest-levels-imbalanced.PNG "imbalance-petur-einarsson")

Summary statistics before cleaning, showed that outliers were present in location data and price data. This finding is highlighted in red. 

These outliers were then removed:
- Location outliers: New York location boundary was found using coordinates. Anything outside the boundary was removed.
- Price: Top and bottom 1% was removed. Typical right-skewed distribution remained. 

![outliers-petur-einarsson](https://github.com/Hunang/Kaggle-RentHop/blob/master/Images/outliers.PNG "outliers-petur-einarsson")
![price-petur-einarsson](https://github.com/Hunang/Kaggle-RentHop/blob/master/Images/price-clean.png "price-petur-einarsson")
![location-petur-einarsson](https://github.com/Hunang/Kaggle-RentHop/blob/master/Images/distribution-apartments.PNG "location-petur-einarsson")

## Features Engineering
Feature engineering involves transforming and combining features in an attempt to produce better representations of the dataset for the purposes of modeling

Created features:
- Broker quality (high interest apartments / total managed apartments)
- Apartment quality
- Price per room
- Number of photos
- Common key features from feature column
- Neighborhood (Manhattan, Queens, Staten Island, etc.)
- Time features (weekday, hour, day)

# Modelling and Inference
The general approach to modelling involved application of a number of different algorithms with the full complement of engineered features. Models selected included Logistic Regression, Random Forest (decision tree ensemble), and Neural Networks. This followed the notion that with a rich enough feature representation only a simple model is required (e.g. Logistic Regression), however in lieu of good features, utilising an algorithm such as neural nets can potentially compensate for this.

## Feature Selection
The relative importance of each feature was used to determine its potential usefulness in modelling as some methods are sensitive to the number of features included. 

This was generated using the Extra-Trees Classifier which finds the mean importance of each feature over the number of trees used in the model (1000 trees were used in this instance). It was decided any feature below a threshold of 0.04 mean importance would be discarded from the modelling process.

![feature-importance-petur-einarsson](https://github.com/Hunang/Kaggle-RentHop/blob/master/Images/feature-importance.PNG "feature-importance-petur-einarsson")

## Algorithm comparison
Aside from including better features, the other primary method of improving prediction was to compare various types of classification models and optimise their associated parameters. In each case, a number of parameters were optimised based on a range using 5-fold cross validation to minimise the risk of overfitting to the training dataset

The models that were selected for comparison included:
- Random Forest Classifier
- Neural Network
- Logistic Regression

The best performing classifier was the Neural Network

#image
![models-petur-einarsson](https://github.com/Hunang/Kaggle-RentHop/blob/master/Images/model-comparison.PNG "models-petur-einarsson")

# Conclusions 
The main deliverable for this project was the predictive model. While in some instances it performed well the primary drawback was the poor classification quality for the high interest apartments.

However, the neural network was quite accurate in predicting low interest apartments and the raw probabilities provide better insight than the absolute classification scores. 

This could be used by Two Sigma to help the landlords that post to the site, that their apartment listing is showing behaviours of an apartment that receives low interest. It could then highlight the features that had the largest impact on the prediction to make recommendations how the landlord could improve their listing, and drive actionable decisions.

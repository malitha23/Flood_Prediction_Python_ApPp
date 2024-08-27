
import pandas as pd

 # Load the dataset
 flood_train = pd.read_csv('train.csv')

 # Display the first few rows and basic information about the dataset
 print(flood_train.head(10))
 print(flood_train.shape)
 print(flood_train.info())

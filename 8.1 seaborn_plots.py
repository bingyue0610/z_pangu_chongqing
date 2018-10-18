import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import keras

model = keras.Sequential()
keras.layers.Dense()
def read_data():
    columns = ['ID', 'casenumber', 'date', 'block', 'iucr', 'primarytypedescription', 'locationdescription', 'arrest',
               'domestic', 'beat', 'districtward', 'communityarea', 'fbicode', 'xcoordinate', 'ycoordinate', 'year',
               'updatedon', 'latitude', 'longitude', 'location']
    tmp_df = pd.read_csv('dataset.csv')
    return tmp_df


# countplot
sns.set(style="darkgrid")
titanic = sns.load_dataset("titanic")
ax = sns.countplot(x="class", data=titanic)


# barplot
sns.set_style("whitegrid")
tips = sns.load_dataset("tips")
ax = sns.barplot(x="day", y="total_bill", data=tips,ci=0)

if __name__ == '__main__':
    df0 = read_data()
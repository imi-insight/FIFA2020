# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 18:11:26 2020

@author: SHATADRU
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.offline import plot
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split


sns.set(style="whitegrid", color_codes=True, font_scale=1.3)



df = pd.read_csv("FIFA_20.csv")
df.head()

df.drop(df.iloc[:, 78:104], inplace = True, axis = 1) 

df = df.drop(['sofifa_id', 'player_url', 'long_name', 'dob', 'real_face',
         'player_positions', 'work_rate', 'player_tags', 'loaned_from',
         'joined', 'contract_valid_until', 'nation_position',
         'nation_jersey_number', 'player_traits'], axis = 1)


sub_count = 0

for i in range(len(df)):
  if (df.loc[i, 'team_position'] == 'SUB'):
        sub_count += 1
        # print(df.loc[i, "short_name"], df.loc[i, "club"], df.loc[i, 'team_position'])
        

print('Total subs :', sub_count)

# Dropping subs to analyze only starting 11 team players
starting_eleven_df = df[df['team_position'] != 'SUB']

print(df.shape)
print(starting_eleven_df.shape)

starting_eleven_df = starting_eleven_df.reset_index(drop = True)

bayern_starting_eleven_df = pd.DataFrame()

print('The Bayern München starting 11 :\n')

for i in range(len(starting_eleven_df)) :
    if (starting_eleven_df.loc[i, 'club'] == 'FC Bayern München'):
        print(starting_eleven_df.loc[i, "short_name"], starting_eleven_df.loc[i, "team_position"])
        bayern_starting_eleven_df = bayern_starting_eleven_df.append(starting_eleven_df.loc[i], ignore_index = True)
        
bayern_starting_eleven_df = bayern_starting_eleven_df.reset_index(drop = True)
bayern_starting_eleven_df.shape

for i in range(len(bayern_starting_eleven_df)) :
    print(bayern_starting_eleven_df.loc[i, 'short_name'], bayern_starting_eleven_df.loc[i, "team_position"])
#joint plots  
sns.jointplot(x = starting_eleven_df['age'], y = starting_eleven_df['potential'],
              joint_kws = {'alpha':0.1,'s':5,'color':'red'},
              marginal_kws = {'color':'red'})

sns.jointplot(x = bayern_starting_eleven_df['age'], y = bayern_starting_eleven_df['potential'],
              joint_kws = {'alpha':0.8,'s':15,'color':'red'},
              marginal_kws = {'color':'red'})
#attribute graphs
player_features = (
    'pace', 'shooting', 'passing', 
    'dribbling', 'defending', 
    'gk_diving', 'gk_handling', 'gk_kicking', 
    'gk_reflexes', 'gk_speed', 'gk_positioning')

from math import pi

idx = 1
plt.figure(figsize=(15,45))

for position_name, features in starting_eleven_df.groupby(
    starting_eleven_df['team_position'])[player_features].mean().iterrows():
   
    top_features = dict(features.nlargest(5))
    
    # number of variables
    categories = top_features.keys()
    N = len(categories)
    
    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values = list(top_features.values())
    values += values[:1]
    
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the plot
    ax = plt.subplot(10, 3, idx, polar=True)

    # Draw one axe per variable + add labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([25,50,75], ["25","50","75"], color="grey", size=7)
    plt.ylim(0,100)
    
    plt.subplots_adjust(hspace = 0.5)
    
    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    
    # Fill area
    ax.fill(angles, values, 'b', alpha=0.1)
    
    plt.title(position_name, size=11, y=1.1)
    
    idx += 1
    
idx = 1
plt.figure(figsize=(15,45))

for position_name, features in bayern_starting_eleven_df.groupby(
    bayern_starting_eleven_df['team_position'])[player_features].mean().iterrows():
   
    top_features = dict(features.nlargest(5))
    
    # number of variables
    categories = top_features.keys()
    N = len(categories)
    
    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values = list(top_features.values())
    values += values[:1]
    
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the plot
    ax = plt.subplot(10, 3, idx, polar=True)

    # Draw one axe per variable + add labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([25,50,75], ["25","50","75"], color="grey", size=7)
    plt.ylim(0,100)
    
    plt.subplots_adjust(hspace = 0.5)
    
    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    
    # Fill area
    ax.fill(angles, values, 'b', alpha=0.1)
    
    plt.title(position_name, size=11, y=1.1)
    
    idx += 1
#pairplots   
cols =['overall','potential','movement_acceleration','mentality_aggression','movement_agility',
       'power_stamina','power_strength','preferred_foot']

df_small = starting_eleven_df[cols]
df_bayern = bayern_starting_eleven_df[cols]

sns.pairplot(df_small, hue ='preferred_foot', palette=['black', 'red'], plot_kws=dict(s=50, alpha =0.8), markers=['^','v'])

sns.pairplot(df_bayern, hue ='preferred_foot', palette=['black', 'red'], plot_kws=dict(s=50, alpha =0.8), markers=['^','v'])


#The predictive analysis of different teams
bayern_count = 0

bayern_df = pd.DataFrame()

for i in range(len(df)) :
    if (df.loc[i, 'club'] == 'FC Bayern München'):
        bayern_count += 1
        print(df.loc[i, "short_name"], df.loc[i, "club"])
        bayern_df = bayern_df.append(df.loc[i], ignore_index = True)
        
print('\nNumber of Bayern players found :', bayern_count)

for i in range(len(bayern_df)) :
    print(bayern_df.loc[i, 'short_name'])
    

barca_count = 0

barca_df = pd.DataFrame()

for i in range(len(df)) :
    if (df.loc[i, 'club'] == 'FC Barcelona'):
        barca_count += 1
        print(df.loc[i, "short_name"], df.loc[i, "club"])
        barca_df = barca_df.append(df.loc[i], ignore_index = True)
        
print('\nNumber of Barca players found :', barca_count)

for i in range(len(barca_df)) :
    print(barca_df.loc[i, 'short_name'])
    

mci_count = 0

mci_df = pd.DataFrame()

for i in range(len(df)) :
    if (df.loc[i, 'club'] == 'Manchester City'):
        mci_count += 1
        print(df.loc[i, "short_name"], df.loc[i, "club"])
        mci_df = mci_df.append(df.loc[i], ignore_index = True)
        
print('\nNumber of Man City players found :', mci_count)

for i in range(len(mci_df)) :
    print(mci_df.loc[i, 'short_name'])
    

rma_count = 0

rma_df = pd.DataFrame()

for i in range(len(df)) :
    if (df.loc[i, 'club'] == 'Real Madrid'):
        rma_count += 1
        print(df.loc[i, "short_name"], df.loc[i, "club"])
        rma_df = rma_df.append(df.loc[i], ignore_index = True)
        
print('\nNumber of Real Madrid players found :', rma_count)

for i in range(len(rma_df)) :
    print(rma_df.loc[i, 'short_name'])

juve_count = 0

juve_df = pd.DataFrame()

for i in range(len(df)) :
    if (df.loc[i, 'club'] == 'Juventus'):
        juve_count += 1
        print(df.loc[i, "short_name"], df.loc[i, "club"])
        juve_df = juve_df.append(df.loc[i], ignore_index = True)
        
print('\nNumber of Juventus players found :', juve_count)

for i in range(len(juve_df)) :
    print(juve_df.loc[i, 'short_name'])

psg_count = 0

psg_df = pd.DataFrame()

for i in range(len(df)) :
    if (df.loc[i, 'club'] == 'Paris Saint-Germain'):
        psg_count += 1
        print(df.loc[i, "short_name"], df.loc[i, "club"])
        psg_df = psg_df.append(df.loc[i], ignore_index = True)
        
print('\nNumber of Paris Saint-Germain players found :', psg_count)

for i in range(len(psg_df)) :
    print(psg_df.loc[i, 'short_name'])

liv_count = 0

liv_df = pd.DataFrame()

for i in range(len(df)) :
    if (df.loc[i, 'club'] == 'Liverpool'):
        liv_count += 1
        print(df.loc[i, "short_name"], df.loc[i, "club"])
        liv_df = liv_df.append(df.loc[i], ignore_index = True)
        
print('\nNumber of Liverpool players found :', liv_count)

for i in range(len(liv_df)) :
    print(liv_df.loc[i, 'short_name'])
    
inter_count = 0

inter_df = pd.DataFrame()

for i in range(len(df)) :
    if (df.loc[i, 'club'] == 'Inter'):
        inter_count += 1
        print(df.loc[i, "short_name"], df.loc[i, "club"])
        inter_df = inter_df.append(df.loc[i], ignore_index = True)
        
print('\nNumber of Inter players found :', inter_count)

for i in range(len(inter_df)) :
    print(inter_df.loc[i, 'short_name'])
# We use the decision classifier individually to test and predict each of the 8 teams
#Defining the dependent and independent variables
x_train = df[['pace','shooting','dribbling','passing','physic','defending']]
x_test = bayern_df[['pace','shooting','dribbling','passing','physic','defending']]
y_train = df[['overall']]
y_test = bayern_df[['overall']]
#Splitting the dataset into train and test data

#x_train, x_test, y_train, y_test =  train_test_split(x,y,test_size = 0.25, random_state= 0)




#Fitting the model
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
print(model)


# Create Decision Tree classifer object
classifier = DecisionTreeClassifier(criterion="entropy",max_depth=16)
# Train Decision Tree Classifer
classifier = classifier.fit(x_train,y_train)
#Predict the response for test dataset
y_pred = classifier.predict(x_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# make predictions
expected = y_test
predicted = classifier.predict(x_test)
# summarize the fit of the model

#print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


#3-d line plot
fig = px.line_3d(bayern_df, x="overall", y="height_cm", z="weight_kg", color='nationality')
plot(fig)


#3-d comparison plots
# Data for a three-dimensional line
zline = bayern_df['age']
xline = bayern_df['overall']
yline = bayern_df['potential']
ax.plot3D(xline, yline, zline, 'purple')

# Data for three-dimensional scattered points
zdata = bayern_df['age']
xdata = bayern_df['overall']
ydata = bayern_df['potential']
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='brg');

# Data for a three-dimensional line
zline = df['age']
xline = df['overall']
yline = df['potential']
ax.plot3D(xline, yline, zline, 'gray')

# Data for three-dimensional scattered points
zdata = df['age']
xdata = df['overall']
ydata = df['potential']
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='red');

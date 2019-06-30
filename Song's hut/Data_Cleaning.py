# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:28:33 2019

@author: Nitesh Sachdev
"""

#data Cleaning

import numpy as np
import pandas as pd
song_dataset=pd.read_csv("Million_song_dataset.csv")
billboard=pd.read_csv("hit_song.csv")
billboard=pd.DataFrame(billboard['Song Title'].drop_duplicates(keep='first')).reset_index()

for col in song_dataset:
    if song_dataset[col].dtype == object:
        song_dataset[col] = song_dataset[col].str.replace("b'","")
        song_dataset[col] = song_dataset[col].str.replace("'","")
        song_dataset[col] = song_dataset[col].str.replace('b"',"")
        song_dataset[col] = song_dataset[col].str.replace('"',"")
for row in range(10000):
    if song_dataset['title'][row].find('(')>0:
        song_dataset['title'][row]=song_dataset['title'][row][:(song_dataset['title'][row].find('(')-1)]

#clean Billboard
        
billboard['Song Title'] = billboard['Song Title'].str.replace("'","")
for row in range(5399):
    if billboard['Song Title'][row].find('(')>0:
        billboard['Song Title'][row] = billboard['Song Title'][row][:(billboard['Song Title'][row].find('(')-1)]



#Matching Dataset and billboard

#import pandas as pd
#dataset=pd.read_csv("Million_song_dataset.csv")
#billboard=pd.read_csv("hit_song.csv")

bill_list=list(billboard["Song Title"])
for song in range(10000):
    if (song_dataset['song_hotttnesss'][song] == 0) or (str(song_dataset['song_hotttnesss'][song])=='nan'):
        if song_dataset['title'][song] in bill_list:
            song_dataset['song_hotttnesss'][song] = 1.0
        else:
            song_dataset['song_hotttnesss'][song] = 0.0
#main_dataset['hit'].value_counts()
            

for song in range (10000):
    if song_dataset['song_hotttnesss'][song]>=0.5:
        song_dataset['song_hotttnesss'][song]=1
    else:
        song_dataset['song_hotttnesss'][song]=0


for col in song_dataset:
    if song_dataset[col].dtype == object:
        song_dataset[col] = song_dataset[col].str.replace("[","")
        song_dataset[col] = song_dataset[col].str.replace("]","")


song_dataset=song_dataset.drop('Unnamed: 0',axis=1)

for year in range(10000):
    if song_dataset['year'][year]==0.0:
        song_dataset['year'][year]='nan'
        
song_dataset['year'] = song_dataset['year'].ffill().bfill()


song_dataset.to_csv("Clean_MSD.csv")
 
    

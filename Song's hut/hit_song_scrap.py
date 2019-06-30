# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:28:33 2019

@author: Nitesh Sachdev
"""


from bs4 import BeautifulSoup
import requests
#import urllib

A=[]
B=[]
C=[]
    

year=1947


while year < 2013:
    #specify the url
    wiki = "http://www.bobborst.com/popculture/top-100-songs-of-the-year/?year="+str(year)
    source = requests.get(wiki).text
    soup = BeautifulSoup(source,"lxml")
    right_table=soup.find('table', class_='sortable alternate songtable')
   
    for row in right_table.findAll('tr'):
        cells = row.findAll('td')
        if len(cells) == 3:
            A.append(cells[1].text.strip())
            B.append(cells[2].text.strip())
            C.append(year)
    
    year=year+1
    print(year)


import pandas as pd
from collections import OrderedDict

col_name = ["Artist","Song Title","Year"]
col_data = OrderedDict(zip(col_name,[A,B,C]))
df = pd.DataFrame(col_data) 
df.to_csv("hit_song.csv")




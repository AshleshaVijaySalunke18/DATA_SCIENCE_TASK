# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 13:28:01 2025

@author: Ashlesha
"""

import requests
from bs4 import BeautifulSoup

link = "https://www.kkwagh.edu.in/engineering/" 

# Send a GET request
response = requests.get(link)
soup = BeautifulSoup(response.content, 'html.parser')

# Redundant line (you already have response)
page = requests.get(link)
page                                            
page.content                                           

# Print formatted HTML
soup.prettify()

# Attempt to extract all <p> tags with class 
title = soup.find_all('p', class_="modal fade show")
# Extract text from each found tag
review_title = []                               
for i in range(0, len(title)):                 
    review_title.append(title[i].get_text())  

# Output the results
review_title                                
print(len(review_title))

#Borad Room

board=soup.find_all('div',class_ ="course_listing")  
board
room=[]                                     
for i in range(0,len(room)):                
    room.append(room[i].get_text())        
room
len(room)  

# ug
ug_pg_sections = soup.find_all("div", class_="header-program-list")

ug_programs = []
pg_programs = []
 
board = soup.find_all('div', class_='program-heading') 
room = []
for tag in board:
    room.append(tag.get_text(strip=True))
room
len(room)

#sport
sp=soup.find_all('div',class_ ="table-responsive table-type-1")  
sp
st=[]                                     
for i in range(0,len(st)):                
    st.append(st[i].get_text())        
st
len(st)  

#activity
sp=soup.find_all('div',class_ ="activity-section pb-0")  
sp
st=[]                                     
for i in range(0,len(st)):                
    st.append(st[i].get_text())        
st
len(st)  

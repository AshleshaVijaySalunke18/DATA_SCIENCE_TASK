# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 12:24:43 2025

@author: Ashlesha
"""

import requests
from bs4 import BeautifulSoup

link = "https://mityeola.com/" 

# Send a GET request
response = requests.get(link)
soup = BeautifulSoup(response.content, 'html.parser')

# Redundant line (you already have response)
page = requests.get(link)
page                                            
page.content                                           

# Print formatted HTML
soup.prettify()

# Attempt to extract all <p> tags with class "ttl1 "
title = soup.find_all('p', class_="elementor-widget-container")
# Extract text from each found tag
review_title = []                               
for i in range(0, len(title)):                 
    review_title.append(title[i].get_text())  

# Output the results
review_title                                
print(len(review_title))

#Borad Room

board=soup.find_all('div',class_ ="elementor-widget-wrap elementor-element-populated")  
board
room=[]                                     
for i in range(0,len(room)):                
    room.append(room[i].get_text())        
room
len(room)  

#Placement 
#Computer Engg 
placement1=soup.find_all('div',class_="elementor-section elementor-top-section elementor-element elementor-element-1d22a9c elementor-section-boxed elementor-section-height-default elementor-section-height-default")  
placement1
place=[]                                     
for i in range(0,len(placement1)):                
    place.append(placement1[i].get_text())        
place
len(place)  

#IT Engg
placement2=soup.find_all('div',class_='elementor-widget-container')
placement2
pla=[]
for i in range(0, len(placement2)):
    pla.append(placement2[i].get_text())
pla
len(pla) 
# Admissions
Admissions=soup.find_all('div',class_='elementor-widget-wrap elementor-element-populated')
Admissions
Admissions1=[]
for i in range(0, len( Admissions)):
    Admissions1.append( Admissions[i].get_text())
Admissions1
len( Admissions1) 
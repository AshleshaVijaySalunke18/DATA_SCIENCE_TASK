# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 20:52:10 2025

@author: Ashlesha
"""


import requests
from bs4 import BeautifulSoup

link = "https://pravaraengg.org.in/" 

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
title = soup.find_all('p', class_="col-2 d-block d-sm-none")      

# Extract text from each found tag
review_title = []                               
for i in range(0, len(title)):                 
    review_title.append(title[i].get_text())  

# Output the results
review_title                                
print(len(review_title))

#navbar

navbar = soup.find('nav', attrs={'class': ['navbar', 'navbar-expand-lg', 'navbar-light', 'bg-light']})
navbar
Deparment=[]                                     
for i in range(0,len(navbar)):                
    Deparment.append(navbar[i].get_text())        
Deparment
len(Deparment)    

#Student life

stud=soup.find_all('div',class_ ="bi bi-chevron-right")  
stud
sport=[]                                     
for i in range(0,len(stud)):                
    sport.append(stud[i].get_text())        
sport
len(sport)  

#Placement 
#Computer Engg 
placement1=soup.find_all('div',class_="w3-image w3-padding w3-round-medium ")  
placement1
place=[]                                     
for i in range(0,len(placement1)):                
    place.append(placement1[i].get_text())        
place
len(place)  

#IT Engg
placement2=soup.find_all('div',class_='spinner')
placement2
pla=[]
for i in range(0, len(placement2)):
    pla.append(placement2[i].get_text())
pla
len(pla) 
# Admissions
Admissions=soup.find_all('div',class_='shadow-lg p-3 mb-5 bg-black')
Admissions
Admissions1=[]
for i in range(0, len( Admissions)):
    Admissions1.append( Admissions[i].get_text())
Admissions1
len( Admissions1) 
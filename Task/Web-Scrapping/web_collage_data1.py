# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 12:14:36 2025
@author: Ashlesha
"""

import requests
from bs4 import BeautifulSoup

link = "https://sndcoe.ac.in/" 

# Send a GET request
response = requests.get(link)
soup = BeautifulSoup(response.content, 'html.parser')

# Redundant line (you already have response)
page = requests.get(link)
page                                            
page.content                                           

# Print formatted HTML
soup.prettify()

# Attempt to extract all <p> tags with class "elementor-heading-title elementor-size-default"
title = soup.find_all('p', class_="elementor-heading-title elementor-size-default")      

# Extract text from each found tag
review_title = []                               
for i in range(0, len(title)):                 
    review_title.append(title[i].get_text())  

# Output the results
review_title                                
print(len(review_title))


Graduate=soup.find_all('div',class_='premium-tabs premium-tabs-style-iconbox premium-tabs-horizontal')  
Graduate
Deparment=[]                                     
for i in range(0,len(Graduate)):                
    Deparment.append(Graduate[i].get_text())        
Deparment
len(Deparment)    



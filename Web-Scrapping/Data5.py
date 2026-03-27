# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 14:21:56 2025

@author: Ashlesha
"""

import requests
from bs4 import BeautifulSoup

link="https://www.sandipfoundation.org/"
response = requests.get(link)
soup = BeautifulSoup(response.content, "html.parser")

# ---------- 1. Placements ----------
url1 = "https://www.sandipfoundation.org/placement/place-student.php"
res1 = requests.get(url1)
soup1 = BeautifulSoup(res1.content, "html.parser")
rows = soup1.select("table tr")[1:6]  
for row in rows:
    data = [td.get_text(strip=True) for td in row.find_all("td")]
data

# ---------- 2. Cources ----------

for a_tag in soup.find_all('a'):
    text = a_tag.get_text(strip=True)
    if any(course in text for course in ['B.Tech', 'M.Tech', 'MBA', 'Engineering']):
        print("-", text)

# ---------- 3. Facilities ----------
facilities = ["Library", "Hostel", "Transport", "Medical", "Cafeteria", "Wi-Fi", "Gym", "Labs"]
for f in facilities:
    print("-", f)

import re
import time
import requests
import pymysql
from bs4 import BeautifulSoup
from selenium import webdriver

# ===== selenium ===== #

browser = webdriver.Chrome(executable_path="./db/chromedriver")
# browser.maximize=window()
url = "https://play.google.com/store/movies/collection/cluster?clp=0g4XChUKD3RvcHNlbGxpbmdfcGFpZBAHGAQ%3D:S:ANO1ljJvXQM&gsr=ChrSDhcKFQoPdG9wc2VsbGluZ19wYWlkEAcYBA%3D%3D:S:ANO1ljK7jAA&hl=en_US&gl=US"
browser.get(url)

interval = 2 # 2초에 한번씩 스크롤 다운
prev_height = browser.execute_script("return document.body.scrollHeight")

while True:
    # 스크롤 가장 아래로 다운
    browser.execute_script("window.scrollTo(0, document.body.scrollHeight)")

    # 페이지 로딩 대기
    time.sleep(interval)

    # 현시점 문서 높이를 저장
    curr_height = browser.execute_script("return document.body.scrollHeight")

    # 모든 스크롤다운 끝난 경우
    if curr_height == prev_height:
        break
    
    # 이전 문서 높이 초기화
    prev_height = curr_height

print("complete scroll down")


# ===== web scraping ===== #

title_list = []
genre_list = []
overview_list = []
rate_list = []
soup = BeautifulSoup(browser.page_source, "lxml")

# scraping for movie titles & genres data
movies = soup.find_all("div", attrs={"class":"kCSSQe"})
for movie in movies:
    # movie titles
    title = movie.find("div", attrs={"class":"WsMG1c nnK0zc"})
    if title is None:
        continue
    else:
        title_list.append(title.get_text())
    
    # movie genres
    genre = movie.find("div", attrs={"class":"KoLSrc"})
    if genre is None:
        continue
    else:
        genre_list.append(genre.get_text())

# movie overviews
overviews = soup.find_all("div", attrs={"class":"b8cIId f5NCO"})
for overview in overviews:
    ov = overview.a.get_text()
    overview_list.append(ov)

# movie rates
rates = soup.find_all("div", attrs={"class":"pf5lIe"})
for rate in rates:
    p = re.compile(r".\..")
    m = p.search(str(rate))
    rate_list.append(m.group())

# make a tuple to insert data in order into the db created
sql_row = [(t, g, o, r) for t, g, o, r in zip(title_list, genre_list, overview_list, rate_list)]


# ===== data into db ===== #

# connection & cursor
conn = pymysql.connect(host="localhost",
                       user="root",
                       password='Dhqxlvmfkdla11@',
                       charset='utf8'
                       )
cur = conn.cursor()

# select a database to use
cur.execute("USE rc_googlemovies") 

# get data into the db
try:
    # INSERT
    with cur:
        sql = "INSERT INTO contentsbased (title, genre, overview, rate) VALUES (%s, %s, %s, %s)"
        value = sql_row
        cur.executemany(sql, value)
    
    conn.commit()

finally:
    conn.close()
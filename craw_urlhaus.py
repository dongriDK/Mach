# import requests
# from bs4 import BeautifulSoup as bs4
import re

# URL = "https://urlhaus.abuse.ch/browse/page/"
f = open("./csv.txt", "r")
hoststr = "[a-zA-Z]"
ip = open("urlhaus_ip.txt","a")
ip_ = []
hostname = open("urlhaus_hostname.txt","a")
hostname_ = []
url = open("urlhaus_url.txt","a")
url_ = []

# for i in range(0,50):
#     response = requests.get(URL+str(i))
#     # status = response.status_code
#     text = response.text
#
#     soup = bs4(text, 'html.parser')
#     url_link = soup.select(
#         'tr > td > a'
#     )
for i in f.readlines():
    a = i.split('","')[2]
    if bool(re.search(r'\d', a)) == True: # 숫자포함
        b = a.split("/")[2] # url 전체
        c = b.split(".")[-1] # root domain 확인
        if bool(re.search(r'\d', c)) == True: # ip
            if ":" in b: # port 제거
                b = b.split(":")[0]
            if b not in ip_:
                ip_.append(b)
                ip.write(b + "\n")
        else:
            if b not in hostname_:
                hostname_.append(b)
                hostname.write(b + "\n")
    else:
        if a not in url_:
            url_.append(a)
            url.write(a + "\n")
    print(a)
ip.close()
hostname.close()
url.close()

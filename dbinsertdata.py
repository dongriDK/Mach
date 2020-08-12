#-*- coding:utf-8 -*-
import pymysql
code = ['123','abcd', '4324', '432341']
name = ['가','나','다','라']
location = ['우리집','방콕','파리','김포']
country = ['korea','korea','korea','korea']
tel = ['010-3333-1111','010-2345-1010','010-3040-1312','010-9999-0000']
convi = ['홀','공원','음식','수영장']

conn = pymysql.connect(host='localhost', user='root', password='1234',db='mydb', charset='utf8')

curs = conn.cursor()

sql = "insert into airport values(%s, %s, %s, %s, %s, %s);"
for i in range(0, len(code)):
	curs.execute(sql, (code[i],name[i],location[i],country[i],tel[i],convi[i]))
conn.commit()
conn.close()

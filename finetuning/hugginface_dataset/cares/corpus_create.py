import mysql.connector

'''
  Bibliografy:
    https://www.w3schools.com/python/python_mysql_getstarted.asp
    https://www.w3schools.com/python/python_mysql_select.asp

'''
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="icd10_dx_hackatonnlp"
)

icd10CodeDict = {}
mycursor = mydb.cursor()

mycursor.execute("SELECT dx_code, long_desc FROM `icd10_dx_order_code` LIMIT 5")

myresult = mycursor.fetchall()

for x in myresult:
  code, description = x
  icd10CodeDict[code] = description

print(icd10CodeDict)
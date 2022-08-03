import lineTool
import requests

def line_notify(msg):
    token_key = 'EvULkJ2z0IYgrHu8RtjDUQoFETiISMKpx3a6fnYR46q'
    header = {'Content-Type':'application/x-www-form-urlencoded',"Authorization":'Bearer '+token_key}
    URL = 'https://notify-api.line.me/api/notify'
    payload = {'message':msg}
    res=requests.post(URL,headers=header,data=payload)


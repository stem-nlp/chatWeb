#coding:utf-8
# Filename:hello_world.py
# 验证服务器，并且收到的所有消息都回复'Hello World!'

import werobot
from model import Robot

tfrobot=Robot.Robot()
robot = werobot.WeRoBot(token='woshiyigejiqiren')

robot.config["APP_ID"] = "wx3b849453c0aad198"
robot.config['ENCODING_AES_KEY'] = 'RynAjQVeCtmQgnAMWOsJGJLTZGuc5aAXDBWSBkDRD3c'

@robot.text
def hello(message):
    answer=tfrobot.ask(str(message.content))
    return answer

# 让服务器监听在 0.0.0.0:80
robot.config['HOST'] = '0.0.0.0'
robot.config['PORT'] = 80
robot.run()

#coding:utf-8
# Filename:hello_world.py
# 验证服务器，并且收到的所有消息都回复'Hello World!'

import werobot
# from model import Robot
from model import Robot_new

# tfrobot=Robot.Robot()
tfrobot = Robot_new.Robot()
robot = werobot.WeRoBot(token='woshiyigejiqiren_sdm')

robot.config["APP_ID"] = "wx4da3c8dab68dd776"
robot.config['ENCODING_AES_KEY'] = '03adbb3c8b67b0e1ddf7a6befb4286b6'

@robot.text
def hello(message):
    answer=tfrobot.ask(str(message.content))
    return answer

# 让服务器监听在 0.0.0.0:80
robot.config['HOST'] = '0.0.0.0'
robot.config['PORT'] = 6006
robot.run()

import tornado.ioloop
import tornado.web
from rekognition.web.views import DashboardHandler, TaskHandler, AuthCreateHandler, AuthLoginHandler
from tornado_sqlalchemy import make_session_factory
from tornado.web import StaticFileHandler
import os
import numpy as np

factory = make_session_factory("mysql://ccextractor:redwood32@localhost:3306/rekognition")

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)

def make_app():
	urls = [
		(r"/", DashboardHandler),
		(r"/login", AuthLoginHandler),
		(r"/register", AuthCreateHandler),
		(r"/addtask", TaskHandler),
		(r"/output/(.*)", StaticFileHandler, {'path':"output/"})
	]
	app = tornado.web.Application(urls,
		template_path=os.path.join(fileDir, "rekognition/web/templates"),
		static_path=os.path.join(fileDir, "rekognition/web/assets"),
		session_factory=factory,
		cookie_secret="random")

	return app

if __name__ == "__main__":
	app = make_app()
	app.listen(8888)
	tornado.ioloop.IOLoop.current().start()
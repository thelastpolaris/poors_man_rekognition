import tornado.ioloop
import tornado.web
from views import DashboardHandler, Upload
import os

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)

def make_app():
	urls = [ 
		(r"/", DashboardHandler), 
		(r"/upload", Upload)
	]
	app = tornado.web.Application(urls,
		template_path=os.path.join(fileDir, "templates"),
		static_path=os.path.join(fileDir, "assets"))

	return app

if __name__ == "__main__":
	app = make_app()
	app.listen(8888)
	tornado.ioloop.IOLoop.current().start()
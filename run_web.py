import tornado.ioloop
import tornado.web
from rekognition.web.views.auth import AuthCreateHandler, AuthLoginHandler, GoogleOAuth2LoginHandler, GithubLoginHandler
from rekognition.web.views.dashboard import DashboardHandler, TaskHandler
from tornado_sqlalchemy import make_session_factory
from tornado.web import StaticFileHandler
import os

factory = make_session_factory("mysql://ccextractor:redwood32@localhost:3306/rekognition")

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)

def make_app():
	urls = [
		(r"/", DashboardHandler),
		(r"/login", AuthLoginHandler),
		(r"/register", AuthCreateHandler),
		(r"/login_google", GoogleOAuth2LoginHandler),
		(r"/login_github", GithubLoginHandler),
		(r"/addtask", TaskHandler),
		(r"/output/(.*)", StaticFileHandler, {'path':"output/"})
	]
	settings = {
		'template_path': os.path.join(fileDir, "rekognition/web/templates"),
		'static_path': os.path.join(fileDir, "rekognition/web/assets"),
		'cookie_secret': 'random',
		'session_factory': factory,
		'xsrf_cookies': True,
		'debug': True,
		'google_oauth': {
			'key': os.environ["GOOGLE_KEY"],
			'secret': os.environ["GOOGLE_SECRET"],
			'redirect_uri': 'http://f7d526a2.ngrok.io/login_google',
			'scope': ['openid', 'email', 'profile']
		},
		'github_oauth': {
			'key': os.environ["GITHUB_KEY"],
			'secret': os.environ["GITHUB_SECRET"],
			'redirect_uri': 'http://f7d526a2.ngrok.io/login_github',
			'scope': ['openid', 'email', 'profile']
		},
		'login_url': "/login"
	}
	app = tornado.web.Application(urls, **settings)

	return app

if __name__ == "__main__":
	app = make_app()
	app.listen(8888)
	tornado.ioloop.IOLoop.current().start()
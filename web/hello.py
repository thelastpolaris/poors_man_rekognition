import tornado.ioloop
import tornado.web

class MainHandler(tornado.web.RequestHandler):
	def get(self):
		self.write('<html><body><form action="/myform" method="POST">'
					'<input type="text" name="message">'
					'<input type="submit" value="Submit">'
					'</form></body></html>')

	def post(self):
		self.set_header("Content-Type", "text/plain")
		self.write("You wrote " + self.get_body_argument("message"))
	# def get(self):
		# self.render("templates/index.html", title="Poor's Man Rekognition - Dashboard")

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/myform", MainHandler)
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()
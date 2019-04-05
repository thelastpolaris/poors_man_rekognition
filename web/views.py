import tornado.ioloop
import tornado.web
import os, uuid
from .rekognition.pipeline.pipeline import Pipeline

__UPLOADS__ = "uploads/"

class DashboardHandler(tornado.web.RequestHandler):
	def get(self):
		self.render("index.html", title = "Poor's Man Rekognition - Dashboard")

	def post(self):
		self.set_header("Content-Type", "text/plain")
		self.write("You wrote " + self.get_body_argument("message"))

class Upload(tornado.web.RequestHandler):
    def post(self):
        fileinfo = self.request.files['filearg'][0]
        # print("fileinfo is", fileinfo)
        fname = fileinfo['filename']
        extn = os.path.splitext(fname)[1]
        cname = str(uuid.uuid4()) + extn
        fh = open(__UPLOADS__ + cname, 'wb')
        fh.write(fileinfo['body'])
        self.finish(cname + " is uploaded!! Check %s folder" %__UPLOADS__)

        # Pipeline()


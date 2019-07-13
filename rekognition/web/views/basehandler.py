from tornado.web import RequestHandler

class BaseHandler(RequestHandler):
    def get_current_user(self):
        return self.get_secure_cookie("user_id")
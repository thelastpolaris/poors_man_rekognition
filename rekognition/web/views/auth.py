from .basehandler import BaseHandler
from tornado.auth import GoogleOAuth2Mixin
from tornado_sqlalchemy import SessionMixin
import tornado.web
import json
import bcrypt

from ..models import User

__UPLOADS__ = "uploads/"

class PMRAuth(SessionMixin, BaseHandler):
    def check_login(self, login, password, password_string = "Password"):
        with self.make_session() as session:
            user = session.query(User).filter(User.login == login).first()
            if user:
                hashed_pass = user.password.encode("utf-8")

                if bcrypt.hashpw(password, hashed_pass) == hashed_pass:
                    self.set_secure_cookie("user_id", str(user.id))
                    self.redirect("/")
                else:
                    self.render("login.html", title=self.title, error="Incorrect {}".format(password_string))
            else:
                self.render("login.html", title=self.title, error="User Not Found")

class AuthCreateHandler(PMRAuth):
    title = "Poor's Man Rekognition - Register"

    def get(self):
        if self.get_current_user():
            self.redirect('/')
            return

        self.render("register.html", title = self.title)

    async def post(self):
        login = self.get_argument("login")
        password = self.get_argument("password").encode("utf-8")

        salt = bcrypt.gensalt()
        password_hashed = bcrypt.hashpw(password, salt)

        with self.make_session() as session:
            user = User(login, password_hashed)
            session.add(user)
            session.commit()

            self.set_secure_cookie("user_id", str(user.id))
            self.redirect("/")


class AuthLoginHandler(PMRAuth):
    title = "Poor's Man Rekognition - Login"

    async def get(self):
        if self.get_current_user():
            self.redirect('/')
            return

        self.render("login.html", title = self.title, error= None)

    async def post(self):
        login = self.get_argument("login")
        password = self.get_argument("password").encode("utf-8")

        self.check_login(login, password)

class GoogleOAuth2LoginHandler(PMRAuth, GoogleOAuth2Mixin):
    def auth_failed(self):
        self.clear_all_cookies()
        raise tornado.web.HTTPError(500, 'Google authentication failed')

    async def get(self):
        if self.get_current_user():
            self.redirect('/')
            return

        settings = self.settings['google_oauth']

        if self.get_argument('code', ""):
            user = await self.get_authenticated_user(redirect_uri=settings["redirect_uri"],
                code=self.get_argument('code'))

            if not user:
                self.auth_failed()

            # Get User Information
            access_token = str(user['access_token'])
            http_client = self.get_auth_http_client()
            response = await http_client.fetch('https://www.googleapis.com/oauth2/v1/userinfo?access_token=' + access_token)

            if not response:
                self.auth_failed()

            user = json.loads(response.body)
            self.handle_user(user["id"].encode("utf-8"), user["email"], user["picture"])

        else:
            self.authorize_redirect(
                redirect_uri=settings["redirect_uri"],
                client_id=settings["key"],
                scope= ['email'],
                response_type='code',
                extra_params={'approval_prompt': 'auto'})

    def handle_user(self, user_id, login, picture=None):
        with self.make_session() as session:
            user = session.query(User).filter(User.login == login).first()

            if not user:
                salt = bcrypt.gensalt()
                user_id_hashed = bcrypt.hashpw(user_id, salt)

                user = User(login, user_id_hashed, picture)
                session.add(user)
                session.commit()

        self.check_login(login, user_id)
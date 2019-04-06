import tornado.web
import os, bcrypt
from tornado.web import RequestHandler
from tornado_sqlalchemy import (SessionMixin, as_future, declarative_base,
                                make_session_factory)

from .pipelines import createPipeline
from .models import User, File

__UPLOADS__ = "uploads/"

class DashboardHandler(SessionMixin, RequestHandler):
    async def get(self):
        if len(self.get_arguments("logout")) > 0:
            self.clear_all_cookies()
            self.redirect("/")
            return
        
        user_id = self.get_secure_cookie("user_id")

        if user_id:
            with self.make_session() as session:
                user = session.query(User).filter(User.id == user_id).first()
                files = session.query(File).filter(File.user_id == user_id).all()
                print(files)

                processed_files = session.query(File).filter(File.user_id == user_id).filter(File.status == 2).count()
                inprocess_files = session.query(File).filter(File.user_id == user_id).filter(File.status != 2).count()
                processing_globally = session.query(File).filter(File.status != 2).count()

                args = {
                    "title": "Poor's Man Rekognition - Dashboard", 
                    "user": user,
                    "files": files, 
                    "processed_files": processed_files, 
                    "inprocess_files": inprocess_files,
                    "processing_globally": processing_globally
                    }

                self.render("index.html", **args)
        else:
            self.redirect("/login")

    def post(self):
        self.set_header("Content-Type", "text/plain")
        self.write("You wrote " + self.get_body_argument("message"))

class AuthCreateHandler(SessionMixin, RequestHandler):
    title = "Poor's Man Rekognition - Register"

    def get(self):
        user_id = self.get_secure_cookie("user_id")
        
        if user_id == None:
            self.render("register.html", title = self.title)
        else:
            self.redirect("/")

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


class AuthLoginHandler(SessionMixin, RequestHandler):
    title = "Poor's Man Rekognition - Login"
    async def get(self):
        user_id = self.get_secure_cookie("user_id")
        
        if user_id == None:
            self.render("login.html", title = self.title, error= None)
        else:
            self.redirect("/")

    async def post(self):
        login = self.get_argument("login")
        password = self.get_argument("password").encode("utf-8")

        with self.make_session() as session:
            user = session.query(User).filter(User.login == login).first()
            if user != None:
                hashed_pass = user.password.encode("utf-8")

                if bcrypt.hashpw(password, hashed_pass) == hashed_pass:
                    self.set_secure_cookie("user_id", str(user.id))
                    self.redirect("/")
                else:
                    self.render("login.html", title = self.title, error="incorrect password")
            else:
                self.render("login.html", title = self.title, error="user Not Found")

class TaskHandler(SessionMixin, RequestHandler):
    async def post(self):
        user_id = self.get_secure_cookie("user_id")

        if user_id != None:
            if os.path.exists(__UPLOADS__) != True:
                os.mkdir(__UPLOADS__)

            fileinfo = self.request.files['filearg'][0]
            fname = fileinfo['filename']
            filename = __UPLOADS__ + fname

            fh = open(filename, 'wb')
            fh.write(fileinfo['body'])

            with self.make_session() as session:
                file = File(user_id, fname)
                session.add(file)
                session.commit()
            
                self.redirect("/")

                # Update status to pending
                file.status = 1
                session.commit()
                
                p = createPipeline(filename)
                
                isFinished = await tornado.ioloop.IOLoop.current().run_in_executor(
                    None,
                    p.run,
                )
                
                fname = os.path.splitext(fname)[0]

                if isFinished:
                    file.status = 2
                    file.json_file = fname + "_output.json"
                    file.output_file = fname + "_output.mp4"
                    session.commit()

                # file.
            # with self.make_session() as session:


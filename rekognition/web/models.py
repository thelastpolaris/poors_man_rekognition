from sqlalchemy import BigInteger, Column, String, SmallInteger
from tornado_sqlalchemy import declarative_base
DeclarativeBase = declarative_base()

class User(DeclarativeBase):
    __tablename__ = 'users'

    id = Column(BigInteger, primary_key=True)
    login = Column(String(255), unique=True)
    password = Column(String(255), unique=False)
    picture = Column(String(1024))
    type = Column(String(128))

    def __init__(self, login, password, type, picture=None):
        self.login = login
        self.password = password
        self.picture = picture
        self.type = type

class File(DeclarativeBase):
    __tablename__ = 'files'

    id = Column(BigInteger, primary_key=True)
    user_id = Column(BigInteger, unique=False)
    filename = Column(String(255), unique=False)
    json_file = Column(String(255), unique=False)
    output_file = Column(String(255), unique=False)
    status = Column(SmallInteger, unique=False)

    def __init__(self, user_id, filename):
        self.user_id = user_id
        self.filename = filename
        self.status = 0
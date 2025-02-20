from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin

# Initialize Flask app
app = Flask(__name__)
app.config.from_object('config')

# Initialize database and authentication
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# User Model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    rfid = db.Column(db.String(50), unique=True, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    dob = db.Column(db.Date, nullable=False)
    telephone = db.Column(db.String(15), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    role = db.Column(db.String(10), nullable=False)
    password = db.Column(db.String(256), nullable=False)

# Sensor Data Model
class SensorData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    rfid = db.Column(db.String(50), db.ForeignKey('user.rfid'), nullable=False)
    weight = db.Column(db.Float, nullable=False)
    height = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    ecg = db.Column(db.Text, nullable=False)
    datetime = db.Column(db.DateTime, nullable=False)
    prediction_result = db.Column(db.Text, nullable=True)

# Import routes (authentication, sensors, dashboard)
from routes import auth, sensors, dashboard
app.register_blueprint(auth.bp)
app.register_blueprint(sensors.bp)
app.register_blueprint(dashboard.bp)

if __name__ == '__main__':
    app.run(debug=True)

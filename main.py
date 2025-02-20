from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_wtf.csrf import CSRFProtect
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Change to a strong secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'  # SQLite Database
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
csrf = CSRFProtect(app)
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

    def set_password(self, password):
        self.password = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password, password)

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

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Home Page
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

# User Registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        rfid = request.form['rfid']
        gender = request.form['gender']
        age = int(request.form['age'])
        dob = datetime.strptime(request.form['dob'], '%Y-%m-%d')
        telephone = request.form['telephone']
        email = request.form['email']
        role = request.form['role']
        password = request.form['password']

        # Check for existing user
        if User.query.filter_by(email=email).first() or User.query.filter_by(rfid=rfid).first():
            flash('User already exists. Try logging in.', 'danger')
            return redirect(url_for('register'))

        new_user = User(name=name, rfid=rfid, gender=gender, age=age, dob=dob, telephone=telephone, email=email, role=role)
        new_user.set_password(password)

        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

# User Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'danger')

    return render_template('login.html')

# Logout Route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# User Dashboard
@app.route('/dashboard')
@login_required
def dashboard():
    sensor_data = SensorData.query.filter_by(rfid=current_user.rfid).all()
    return render_template('dashboard.html', sensor_data=sensor_data)

# Add Exercise & Health Data
@app.route('/add_data', methods=['GET', 'POST'])
@login_required
def add_data():
    if request.method == 'POST':
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        ecg = request.form['ecg']
        bmi = weight / (height / 100) ** 2
        date_time = datetime.now()

        new_sensor_data = SensorData(rfid=current_user.rfid, weight=weight, height=height, bmi=bmi, ecg=ecg, datetime=date_time)
        db.session.add(new_sensor_data)
        db.session.commit()
        flash('Data added successfully!', 'success')
        return redirect(url_for('dashboard'))

    return render_template('add_data.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001,debug=True)
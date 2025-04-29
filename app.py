from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from datetime import datetime,timedelta
import re
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
from functools import wraps

# Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_secret_key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# --- Custom Decorator for Admin-Only Access ---
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            flash("You do not have permission to access this page.", "danger")
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# --- Database Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    rfid = db.Column(db.String(50), unique=True, nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    dob = db.Column(db.Date, nullable=False)
    telephone = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    role = db.Column(db.String(50), nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def is_active(self):
        return True

    def is_authenticated(self):
        return True

    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.id)

    def __repr__(self):
        return f"<User {self.name} ({self.email})>"

class SensorData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    rfid = db.Column(db.String(50), db.ForeignKey('user.rfid', ondelete='CASCADE'), nullable=False)
    weight = db.Column(db.Float, nullable=False)
    height = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    ecg = db.Column(db.Text)
    datetime = db.Column(db.DateTime, nullable=False)
    suggested_sport = db.Column(db.String(100))
    user = db.relationship('User', backref=db.backref('sensor_data', lazy=True))

    def __repr__(self):
        return f"<SensorData for RFID: {self.rfid} at {self.datetime}>"

# Create database tables
with app.app_context():
    db.create_all()

# --- Helper Functions ---
def create_user_in_db(name, rfid, gender, age, dob, telephone, email, role, password):
    user = User(name=name, rfid=rfid, gender=gender, age=age, dob=dob, telephone=telephone, email=email, role=role)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return user.id

def get_user_by_id(user_id):
    return User.query.get(user_id)

def get_user_by_rfid(rfid):
    return User.query.filter_by(rfid=rfid).first()

def get_user_by_email(email):
    return User.query.filter_by(email=email).first()

def update_user_in_db(user_id, name=None, rfid=None, gender=None, age=None, dob=None, telephone=None, email=None, role=None, password=None):
    user = get_user_by_id(user_id)
    if user:
        if name: user.name = name
        if rfid: user.rfid = rfid
        if gender: user.gender = gender
        if age is not None: user.age = age
        if dob: user.dob = dob
        if telephone: user.telephone = telephone
        if email: user.email = email
        if role: user.role = role
        if password: user.set_password(password)
        db.session.commit()
        return True
    return False

def delete_user_from_db(user_id):
    user = get_user_by_id(user_id)
    if user:
        db.session.delete(user)
        db.session.commit()
        return True
    return False

def is_valid_email(email):
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))

def is_valid_password(password):
    pattern = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$"
    return bool(re.match(pattern, password))

def is_valid_telephone(telephone):
    pattern = r"^\+?\d{10,15}$"
    return bool(re.match(pattern, telephone))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Load Machine Learning Model ---
try:
    with open('fitness_modelx.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('label_encoderx.pkl', 'rb') as le_file:
        le = pickle.load(le_file)
except FileNotFoundError:
    print("Error: Model or label encoder files not found.")
    exit()

# --- Flask Routes ---
@app.route("/")
def index():
    return render_template('home.html')

@app.route('/users')
@admin_required
def list_users():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    users = User.query.paginate(page=page, per_page=per_page, error_out=False)
    return render_template('index.html', users=users.items, pagination=users)

@app.route('/users/<int:user_id>')
@admin_required
def user_details(user_id):
    user = get_user_by_id(user_id)
    if not user:
        flash("User not found.", "danger")
        return redirect(url_for('list_users'))
    return render_template('user_details.html', user=user)

@app.route('/users/create', methods=['GET', 'POST'])
@login_required
def create_user():
    # Allow creating an admin if no admins exist
    no_admins = User.query.filter_by(role='admin').count() == 0

    # Restrict to admins unless no admins exist
    if not no_admins and current_user.role != 'admin':
        flash("Only admins can create users.", "danger")
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        try:
            name = request.form.get('name').strip()
            rfid = request.form.get('rfid').strip()
            gender = request.form.get('gender').strip()
            age_str = request.form.get('age')
            dob_str = request.form.get('dob')
            telephone = request.form.get('telephone').strip()
            email = request.form.get('email').strip()
            role = request.form.get('role').strip()
            password = request.form.get('password')

            if not all([name, rfid, gender, age_str, dob_str, telephone, email, role, password]):
                flash("All fields are required!", "danger")
                return render_template('create_user.html')

            if get_user_by_rfid(rfid):
                flash("RFID already exists!", "danger")
                return render_template('create_user.html')
            
            if get_user_by_email(email):
                flash("Email already exists!", "danger")
                return render_template('create_user.html')

            try:
                age = int(age_str)
                if age < 0:
                    raise ValueError("Age must be non-negative")
            except ValueError:
                flash("Invalid age", "danger")
                return render_template('create_user.html')

            try:
                dob = datetime.strptime(dob_str, '%Y-%m-%d').date()
            except ValueError:
                flash("Invalid date of birth format (YYYY-MM-DD)", "danger")
                return render_template('create_user.html')

            if not is_valid_email(email):
                flash("Invalid email format", "danger")
                return render_template('create_user.html')

            if not is_valid_password(password):
                flash("Password must be at least 8 characters with uppercase, lowercase, number, and special character.", "danger")
                return render_template('create_user.html')

            if not is_valid_telephone(telephone):
                flash("Invalid telephone number format.", "danger")
                return render_template('create_user.html')

            valid_roles = ['admin', 'user']
            if role not in valid_roles:
                flash("Invalid role selected.", "danger")
                return render_template('create_user.html')

            user_id = create_user_in_db(name, rfid, gender, age, dob, telephone, email, role, password)
            flash("User created successfully!", "success")
            return redirect(url_for('list_users'))

        except Exception as e:
            print(f"An error occurred: {e}")
            flash("An error occurred. Please try again.", "danger")
            return render_template('create_user.html')

    return render_template('create_user.html')

@app.route('/users/<int:user_id>/edit', methods=['GET', 'POST'])
@admin_required
def edit_user(user_id):
    user = get_user_by_id(user_id)
    if not user:
        flash("User not found.", "danger")
        return redirect(url_for('list_users'))

    if request.method == 'POST':
        try:
            name = request.form.get('name').strip()
            rfid = request.form.get('rfid').strip()
            gender = request.form.get('gender').strip()
            age_str = request.form.get('age')
            dob_str = request.form.get('dob')
            telephone = request.form.get('telephone').strip()
            email = request.form.get('email').strip()
            role = request.form.get('role').strip()
            password = request.form.get('password')

            if rfid != user.rfid and get_user_by_rfid(rfid):
                flash("RFID already exists!", "danger")
                return render_template('edit_user.html', user=user)
            
            if email != user.email and get_user_by_email(email):
                flash("Email already exists!", "danger")
                return render_template('edit_user.html', user=user)

            try:
                age = int(age_str) if age_str else user.age
                if age < 0:
                    raise ValueError("Age must be non-negative")
            except ValueError:
                flash("Invalid age", "danger")
                return render_template('edit_user.html', user=user)

            try:
                dob = datetime.strptime(dob_str, '%Y-%m-%d').date() if dob_str else user.dob
            except ValueError:
                flash("Invalid date of birth format (YYYY-MM-DD)", "danger")
                return render_template('edit_user.html', user=user)

            if email and not is_valid_email(email):
                flash("Invalid email format", "danger")
                return render_template('edit_user.html', user=user)

            if password and not is_valid_password(password):
                flash("Password must be at least 8 characters with uppercase, lowercase, number, and special character.", "danger")
                return render_template('edit_user.html', user=user)

            if telephone and not is_valid_telephone(telephone):
                flash("Invalid telephone number format.", "danger")
                return render_template('edit_user.html', user=user)

            valid_roles = ['admin', 'user']
            if role not in valid_roles:
                flash("Invalid role selected.", "danger")
                return render_template('edit_user.html', user=user)

            update_user_in_db(user_id, name=name, rfid=rfid, gender=gender, age=age, dob=dob, 
                            telephone=telephone, email=email, role=role, password=password)
            flash("User updated successfully!", "success")
            return redirect(url_for('user_details', user_id=user_id))

        except Exception as e:
            print(f"An error occurred: {e}")
            flash("An error occurred. Please try again.", "danger")
            return render_template('edit_user.html', user=user)

    return render_template('edit_user.html', user=user)

@app.route('/users/<int:user_id>/delete', methods=['POST'])
@admin_required
def delete_user(user_id):
    if delete_user_from_db(user_id):
        flash("User deleted successfully!", "success")
    else:
        flash("User not found.", "danger")
    return redirect(url_for('list_users'))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password", "danger")
    
    return render_template("login.html")

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully', 'success')
    return redirect(url_for('login'))

@app.route('/add_sensor_data', methods=['GET', 'POST'])
@login_required
def add_sensor_data():
    if request.method == 'POST':
        try:
            weight = float(request.form['weight'])
            height = float(request.form['height'])
            ecg = request.form['ecg']

            bmi = weight / (height / 100) ** 2
            date_time = datetime.now()

            new_sensor_data = SensorData(rfid=current_user.rfid, weight=weight, height=height, bmi=bmi, ecg=ecg, datetime=date_time)
            db.session.add(new_sensor_data)
            db.session.commit()

            flash("Sensor data added successfully!", "success")
            return redirect(url_for('dashboard'))

        except (ValueError, TypeError, KeyError) as e:
            flash(f"Invalid input: {e}", "danger")
            return render_template('add_sensor_data.html')

        except Exception as e:
            print(f"An error occurred: {e}")
            flash("An error occurred. Please try again.", "danger")
            return render_template('add_sensor_data.html')

    return render_template('add_sensor_data.html')

@app.route('/api/sensor_data', methods=['POST'])
@login_required
def add_sensor_data_api():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"message": "No data provided"}), 400

        rfid = data.get('rfid')
        weight = data.get('weight')
        height = data.get('height')
        ecg = data.get('ecg')

        if not all([rfid, weight, height, ecg]):
            return jsonify({"message": "Missing required fields (rfid, weight, height, ecg)"}), 400

        # Ensure the RFID belongs to the current user or the user is an admin
        if rfid != current_user.rfid and current_user.role != 'admin':
            return jsonify({"message": "Unauthorized: Cannot add data for another user"}), 403

        current_user_data = User.query.filter_by(rfid=rfid).first()
        if not current_user_data:
            return jsonify({"message": "User not found"}), 404

        weight = float(weight)
        height = float(height)
        bmi = weight / (height / 100) ** 2
        date_time = datetime.now()

        resting_hr = ecg
        workout_hr = ecg

        input_data = pd.DataFrame([{
            'Age': current_user_data.age,
            'Height (cm)': height,
            'Weight (kg)': weight,
            'Resting HR': resting_hr,
            'Workout HR': workout_hr
        }])

        predicted_label = model.predict(input_data)[0]
        suggested_sport = le.inverse_transform([predicted_label])[0]

        new_sensor_data = SensorData(
            rfid=current_user_data.rfid,
            weight=weight,
            height=height,
            bmi=bmi,
            ecg=ecg,
            suggested_sport=suggested_sport,
            datetime=date_time
        )
        db.session.add(new_sensor_data)
        db.session.commit()

        return jsonify({
            "message": "Sensor data added successfully",
            "rfid": rfid,
            "bmi": round(bmi, 2),
            "ecg": ecg,
            "suggested_sport": suggested_sport
        }), 201

    except (ValueError, TypeError) as e:
        return jsonify({"message": f"Invalid input: {e}"}), 400

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"message": "An error occurred"}), 500

@app.route('/sensor_data')
@login_required
def view_sensor_data():
    try:
        # Admins see all sensor data; users see only their own
        if current_user.role == 'admin':
            sensor_data = SensorData.query.all()
        else:
            sensor_data = SensorData.query.filter_by(rfid=current_user.rfid).all()

        data = [{
            "rfid": entry.rfid,
            "weight": entry.weight,
            "height": entry.height,
            "bmi": round(entry.bmi, 2),
            "ecg": entry.ecg,
            "suggested_sport": entry.suggested_sport,
            "datetime": entry.datetime.strftime("%Y-%m-%d %H:%M:%S")
        } for entry in sensor_data]
        return render_template('sensor_data.html', data=data)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"message": "An error occurred while fetching data."}), 500


@app.route('/plot')
@login_required
def plot():
    users = User.query.all() if current_user.role == 'admin' else []
    return render_template('sensor_data_plot.html', users=users)
@app.route("/dashboard")
@login_required
def dashboard():
    user_data = SensorData.query.filter_by(rfid=current_user.rfid).all()
    return render_template("dashboard.html", user=current_user, sensor_data=user_data)

@app.route('/api/sensor_data', methods=['GET'])
@login_required
def get_sensor_data():
    try:
        # Admins see all sensor data; users see only their own
        if current_user.role == 'admin':
            sensor_data = SensorData.query.all()
        else:
            sensor_data = SensorData.query.filter_by(rfid=current_user.rfid).all()

        sensor_data_list = [{
            "id": data.id,
            "rfid": data.rfid,
            "weight": data.weight,
            "height": data.height,
            "bmi": data.bmi,
            "ecg": data.ecg,
            "suggested_sport": entry.suggested_sport,
            "datetime": data.datetime.strftime("%Y-%m-%d %H:%M:%S")
        } for data in sensor_data]
        return jsonify({"sensor_data": sensor_data_list}), 200
    except Exception as e:
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500



import logging
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, Response
from datetime import datetime, timedelta
from sqlalchemy.exc import SQLAlchemyError
# ... (other imports and existing code remain unchanged)

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

@app.route('/sensor_data/plot', methods=['GET'])
@login_required
def plot_sensor_data():
    try:
        # Get query parameters
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        rfid = request.args.get('rfid')  # For admin filtering by user

        # Default date range: last 30 days
        end_date = datetime.now()
        if end_date_str:
            try:
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            except ValueError:
                logging.error(f"Invalid end_date format: {end_date_str}")
                return jsonify({"message": "Invalid end_date format. Use YYYY-MM-DD."}), 400

        start_date = end_date - timedelta(days=30)
        if start_date_str:
            try:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            except ValueError:
                logging.error(f"Invalid start_date format: {start_date_str}")
                return jsonify({"message": "Invalid start_date format. Use YYYY-MM-DD."}), 400

        # Validate date range
        if start_date > end_date:
            logging.error("start_date is after end_date")
            return jsonify({"message": "Start date cannot be after end date."}), 400

        # Query sensor data based on role
        query = SensorData.query
        if current_user.role != 'admin':
            query = query.filter_by(rfid=current_user.rfid)
        elif rfid:
            if not User.query.filter_by(rfid=rfid).first():
                logging.error(f"Invalid RFID: {rfid}")
                return jsonify({"message": "Invalid RFID."}), 400
            query = query.filter_by(rfid=rfid)

        # Apply date range filter
        sensor_data = query.filter(
            SensorData.datetime >= start_date,
            SensorData.datetime <= end_date + timedelta(days=1)  # Include end date
        ).order_by(SensorData.datetime).all()

        # Log query details
        logging.debug(f"User {current_user.email} queried plot: rfid={rfid or 'all'}, start_date={start_date}, end_date={end_date}, results={len(sensor_data)}")

        # Handle empty data
        if not sensor_data:
            logging.info(f"No sensor data found for user {current_user.email}, rfid={rfid or 'all'}")
            return jsonify({
                "message": "No data available for the selected filters.",
                "dates": [],
                "weights": [],
                "bmis": [],
                "ecgs": [],
                "sports": {"labels": [], "values": []},
                "metrics": {
                    "avg_weight": 0,
                    "avg_bmi": 0,
                    "max_ecg": 0,
                    "min_ecg": 0,
                    "data_points": 0
                }
            }), 200

        # Prepare data for plots
        weights = []
        bmis = []
        ecgs = []
        dates = []
        sports = []

        for entry in sensor_data:
            weights.append(entry.weight)
            bmis.append(entry.bmi)
            try:
                ecg_value = float(entry.ecg) if entry.ecg and entry.ecg.replace('.', '', 1).replace('-', '', 1).isdigit() else 0
            except (ValueError, AttributeError):
                logging.warning(f"Invalid ECG value for entry {entry.id}: {entry.ecg}")
                ecg_value = 0
            ecgs.append(ecg_value)
            dates.append(entry.datetime.strftime("%Y-%m-%d %H:%M:%S"))
            if entry.suggested_sport:
                sports.append(entry.suggested_sport)

        # Calculate additional metrics
        avg_weight = sum(weights) / len(weights) if weights else 0
        avg_bmi = sum(bmis) / len(bmis) if bmis else 0
        max_ecg = max(ecgs) if ecgs else 0
        min_ecg = min(ecgs) if ecgs else 0

        # Count frequency of suggested sports
        sport_counts = {}
        for sport in sports:
            sport_counts[sport] = sport_counts.get(sport, 0) + 1
        sport_labels = list(sport_counts.keys())
        sport_values = list(sport_counts.values())

        return jsonify({
            'dates': dates,
            'weights': weights,
            'bmis': bmis,
            'ecgs': ecgs,
            'sports': {
                'labels': sport_labels,
                'values': sport_values
            },
            'metrics': {
                'avg_weight': round(avg_weight, 2),
                'avg_bmi': round(avg_bmi, 2),
                'max_ecg': round(max_ecg, 2),
                'min_ecg': round(min_ecg, 2),
                'data_points': len(weights)
            }
        })

    except SQLAlchemyError as e:
        logging.error(f"Database error in plot_sensor_data: {str(e)}")
        return jsonify({"message": "Database error occurred."}), 500
    except Exception as e:
        logging.error(f"Unexpected error in plot_sensor_data: {str(e)}")
        return jsonify({"message": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/sensor_data/download', methods=['GET'])
@login_required
def download_sensor_data():
    try:
        # Get query parameters
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        rfid = request.args.get('rfid')

        # Default date range: last 30 days
        end_date = datetime.now()
        if end_date_str:
            try:
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            except ValueError:
                logging.error(f"Invalid end_date format for download: {end_date_str}")
                flash("Invalid end_date format. Use YYYY-MM-DD.", "danger")
                return redirect(url_for('plot'))

        start_date = end_date - timedelta(days=30)
        if start_date_str:
            try:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            except ValueError:
                logging.error(f"Invalid start_date format for download: {start_date_str}")
                flash("Invalid start_date format. Use YYYY-MM-DD.", "danger")
                return redirect(url_for('plot'))

        # Validate date range
        if start_date > end_date:
            logging.error("start_date is after end_date for download")
            flash("Start date cannot be after end date.", "danger")
            return redirect(url_for('plot'))

        # Query sensor data based on role
        query = SensorData.query
        if current_user.role != 'admin':
            query = query.filter_by(rfid=current_user.rfid)
        elif rfid:
            if not User.query.filter_by(rfid=rfid).first():
                logging.error(f"Invalid RFID for download: {rfid}")
                flash("Invalid RFID.", "danger")
                return redirect(url_for('plot'))
            query = query.filter_by(rfid=rfid)

        # Apply date range filter
        sensor_data = query.filter(
            SensorData.datetime >= start_date,
            SensorData.datetime <= end_date + timedelta(days=1)
        ).order_by(SensorData.datetime).all()

        # Log download request
        logging.debug(f"User {current_user.email} downloaded data: rfid={rfid or 'all'}, start_date={start_date}, end_date={end_date}, results={len(sensor_data)}")

        # Handle empty data
        if not sensor_data:
            flash("No data available to download for the selected filters.", "danger")
            return redirect(url_for('plot'))

        # Prepare CSV data
        import csv
        from io import StringIO
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['RFID', 'Weight', 'Height', 'BMI', 'ECG', 'Suggested Sport', 'Datetime'])
        for entry in sensor_data:
            writer.writerow([
                entry.rfid,
                entry.weight,
                entry.height,
                entry.bmi,
                entry.ecg or '',
                entry.suggested_sport or '',
                entry.datetime.strftime("%Y-%m-%d %H:%M:%S")
            ])

        # Return CSV as downloadable file
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={"Content-Disposition": "attachment;filename=sensor_data.csv"}
        )

    except SQLAlchemyError as e:
        logging.error(f"Database error in download_sensor_data: {str(e)}")
        flash("Database error occurred while generating the CSV.", "danger")
        return redirect(url_for('plot'))
    except Exception as e:
        logging.error(f"Unexpected error in download_sensor_data: {str(e)}")
        flash(f"An unexpected error occurred: {str(e)}", "danger")
        return redirect(url_for('plot'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

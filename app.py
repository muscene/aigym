from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import re  # For email validation
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import re
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load your dataset from a CSV file

import pickle
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder# Flask app

from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash


app = Flask(__name__)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

app.secret_key = 'your_secret_key'  # VERY IMPORTANT: Change this!
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# --- Database Model ---
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
        from werkzeug.security import check_password_hash
        return check_password_hash(self.password_hash, password)
    # def check_password(self, password):
    #     return check_password_hash(self.password_hash, password)
    def is_active(self):
        return True  # Always active

    def is_authenticated(self):
        return True  # Assume authenticated

    def is_anonymous(self):
        return False  # Not an anonymous user

    def get_id(self):
        return str(self.id)  # Return ID as a string

    def __repr__(self):
        return f"<User {self.name} ({self.email})>"

    def __repr__(self):
        return '<User %r>' % self.name

with app.app_context():
    db.create_all()
# class SensorData(db.Model):  # New SensorData model
#     id = db.Column(db.Integer, primary_key=True)
#     rfid = db.Column(db.String(50), db.ForeignKey('user.rfid'), nullable=False)  # Foreign key to User
#     weight = db.Column(db.Float, nullable=False)
#     height = db.Column(db.Float, nullable=False)
#     bmi = db.Column(db.Float, nullable=False)
#     ecg = db.Column(db.Text)  # ECG data (could be large, so Text is appropriate)
#     datetime = db.Column(db.DateTime, nullable=False)
#     user = db.relationship('User', backref=db.backref('sensor_data', lazy=True)) # Relationship to User

#     def __repr__(self):
#         return f"<SensorData for RFID: {self.rfid} at {self.datetime}>"
class SensorData(db.Model):  
    id = db.Column(db.Integer, primary_key=True)
    rfid = db.Column(db.String(50), db.ForeignKey('user.rfid'), nullable=False)  # Foreign key to User
    weight = db.Column(db.Float, nullable=False)
    height = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    ecg = db.Column(db.Text)  # ECG data (could be large, so Text is appropriate)
    datetime = db.Column(db.DateTime, nullable=False)
    suggested_sport = db.Column(db.String(100))  # New column to store the predicted sport
    user = db.relationship('User', backref=db.backref('sensor_data', lazy=True))  # Relationship to User

    def __repr__(self):
        return f"<SensorData for RFID: {self.rfid} at {self.datetime}>"
with app.app_context():
    db.create_all()  # Create the tables

# --- Helper Functions ---
def create_user_in_db(name, rfid, gender, age, dob, telephone, email, role, password):
    user = User(name=name, rfid=rfid, gender=gender, age=age, dob=dob, telephone=telephone, email=email, role=role)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return user.id
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

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
        if age: user.age = age
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
    # Basic email validation using regex
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))



try:
    with open('fitness_modelx.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('label_encoderx.pkl', 'rb') as le_file:
        le = pickle.load(le_file)
except FileNotFoundError:
    print("Error: Model or label encoder files not found. Make sure they are in the same directory or provide the correct path.")
    exit()


@app.route("/")
def index():
    return render_template('home.html')

# --- Flask Routes ---
@app.route('/users')
def list_users():
    users = User.query.all()
    return render_template('index.html', users=users)

@app.route('/users/<int:user_id>')
def user_details(user_id):
    user = get_user_by_id(user_id)
    if not user:
        return "User not found", 404
    return render_template('user_details.html', user=user)

@app.route('/users/create', methods=['GET', 'POST'])
def create_user():
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

            if not name or not rfid or not gender or not age_str or not dob_str or not telephone or not email or not role or not password:
                flash("All fields are required!")
                return render_template('create_user.html')

            if get_user_by_rfid(rfid):
                flash("RFID already exists!")
                return render_template('create_user.html')
            
            if get_user_by_email(email):
                flash("Email already exists!")
                return render_template('create_user.html')

            try:
                age = int(age_str)
                if age < 0:
                    raise ValueError("Age must be non-negative")
            except ValueError:
                flash("Invalid age")
                return render_template('create_user.html')

            try:
                dob = datetime.strptime(dob_str, '%Y-%m-%d').date()  # Convert to date object
            except ValueError:
                flash("Invalid date of birth format (YYYY-MM-DD)")
                return render_template('create_user.html')

            if not is_valid_email(email):
                flash("Invalid email format")
                return render_template('create_user.html')
            
            user_id = create_user_in_db(name, rfid, gender, age, dob, telephone, email, role, password)
            flash("User created successfully!")
            return redirect(url_for('list_users'))

        except Exception as e:
            print(f"An error occurred: {e}")  # Log the error for debugging
            flash("An error occurred. Please try again.")  # User-friendly message
            return render_template('create_user.html')

    return render_template('create_user.html')




@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for("dashboard"))  # Redirect to dashboard
        else:
            flash("Invalid email or password", "danger")
    
    return render_template("login.html")
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully', 'success')
    return redirect(url_for('login'))

# ... (rest of the routes - similar error handling as create_user)
# --- Routes ---
@app.route('/add_sensor_data', methods=['GET', 'POST'])  # Important: Add GET
def add_sensor_data():
    # Assuming you have some way to get the current user (e.g., from a login session)
    # Replace this with your actual user retrieval method.
    current_user = User.query.filter_by(rfid='your_current_user_rfid').first()  # Example
    if not current_user:
        flash("User not logged in or not found")
        return redirect(url_for('some_route'))  # Redirect to appropriate page

    if request.method == 'POST':  # Handle POST requests (form submission)
        try:
            weight = float(request.form['weight'])
            height = float(request.form['height'])
            ecg = request.form['ecg']

            bmi = weight / (height / 100) ** 2
            date_time = datetime.now()

            new_sensor_data = SensorData(rfid=current_user.rfid, weight=weight, height=height, bmi=bmi, ecg=ecg, datetime=date_time)
            db.session.add(new_sensor_data)
            db.session.commit()

            flash("Sensor data added successfully!")
            return redirect(url_for('some_route'))  # Redirect to wherever you want

        except (ValueError, TypeError, KeyError) as e:
            flash(f"Invalid input: {e}")
            return redirect(url_for('some_route'))  # Redirect back to the form

        except Exception as e:
            print(f"An error occurred: {e}")  # Log the error
            flash("An error occurred. Please try again.")
            return redirect(url_for('some_route'))

    return render_template('add_sensor_data.html')  # Handle GET requests (display form)
    # @app.route('/api/sensor_data', methods=['POST'])  # API endpoint
    # def add_sensor_data_api():
    #     try:
    #         data = request.get_json()  # Get JSON data from the request body

    #         if not data:
    #             return jsonify({"message": "No data provided"}), 400

    #         rfid = data.get('rfid')  # Get rfid from the JSON data
    #         weight = data.get('weight')
    #         height = data.get('height')
    #         ecg = data.get('ecg')

    #         if not rfid or not weight or not height or not ecg:
    #             return jsonify({"message": "Missing required fields (rfid, weight, height, ecg)"}), 400

    #         current_user = User.query.filter_by(rfid=rfid).first() # user by rfid
    #         if not current_user:
    #             return jsonify({"message": "User not found"}), 404

    #         weight = float(weight)
    #         height = float(height)

    #         bmi = weight / (height / 100) ** 2
    #         date_time = datetime.now()

    #         new_sensor_data = SensorData(rfid=current_user.rfid, weight=weight, height=height, bmi=bmi, ecg=ecg, datetime=date_time)
    #         db.session.add(new_sensor_data)
    #         db.session.commit()

    #         return jsonify({"message": "Sensor data added successfully"}), 201  # 201 Created

    #     except (ValueError, TypeError) as e:
    #         return jsonify({"message": f"Invalid input: {e}"}), 400
@app.route('/api/sensor_data', methods=['POST'])
def add_sensor_data_api():
    try:
        data = request.get_json()  # Get JSON data from request body
        if not data:
            return jsonify({"message": "No data provided"}), 400
        # Extract data from request
        rfid = data.get('rfid')
        weight = data.get('weight')
        height = data.get('height')
        ecg = data.get('ecg')

        if not all([rfid, weight, height, ecg]):
            return jsonify({"message": "Missing required fields (rfid, weight, height, ecg)"}), 400
        # 1️⃣ **Find the user whose rfid matches the sensor data rfid**
        current_user = User.query.filter_by(rfid=rfid).first()
        if not current_user:
            return jsonify({"message": "User not found"}), 404

        # 2️⃣ **Compute BMI**
        weight = float(weight)
        height = float(height)
        bmi = weight / (height / 100) ** 2
        date_time = datetime.now()

        # 3️⃣ **Dynamically set `resting_hr` and `workout_hr` to `ecg` (No DB update)**
        resting_hr = ecg
        workout_hr = ecg

        # 4️⃣ **Prepare input data for prediction**
        input_data = pd.DataFrame([{
            'Age': current_user.age,
            'Height (cm)': height,
            'Weight (kg)': weight,
            'Resting HR': resting_hr,
            'Workout HR': workout_hr
        }])

        # 5️⃣ **Predict Suggested Sport**
        predicted_label = model.predict(input_data)[0]
        suggested_sport = le.inverse_transform([predicted_label])[0]

        # **Debugging: Check if predicted_sport is correct**
        print(f"Predicted sport: {suggested_sport}")

        # 6️⃣ **Save Sensor Data in the Database (Including `suggested_sport`)**
        new_sensor_data = SensorData(
            rfid=current_user.rfid,
            weight=weight,
            height=height,
            bmi=bmi,
            ecg=ecg,
            suggested_sport=suggested_sport,  # Save the predicted sport
            datetime=date_time
        )
        db.session.add(new_sensor_data)
        db.session.commit()

        # Return response with suggested sport
        return jsonify({
            "message": "Sensor data added successfully",
            "rfid": rfid,
            "bmi": round(bmi, 2),
            "ecg": ecg,
            "suggested_sport": suggested_sport
        }), 201  # 201 Created

    except (ValueError, TypeError) as e:
        return jsonify({"message": f"Invalid input: {e}"}), 400

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"message": "An error occurred"}), 500


@app.route('/sensor_data')
def view_sensor_data():
    try:
        # Fetch all sensor data from the database
        sensor_data = SensorData.query.all()
        # Prepare data to send to frontend
        data = []
        for entry in sensor_data:
            data.append({
                "rfid": entry.rfid,
                "weight": entry.weight,
                "height": entry.height,
                "bmi": round(entry.bmi, 2),
                "ecg": entry.ecg,
                "suggested_sport": entry.suggested_sport,
                "datetime": entry.datetime.strftime("%Y-%m-%d %H:%M:%S")
            })

        # Return the rendered HTML template with data
        return render_template('sensor_data.html', data=data)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"message": "An error occurred while fetching data."}), 500

@app.route('/sensor_data/plot')
def plot_sensor_data():
    try:
        # Fetch all sensor data from the database
        sensor_data = SensorData.query.all()

        # Prepare lists for weight, bmi, and datetime
        weights = []
        bmis = []
        ecgis = []
        dates = []

        # Populate the lists with data
        for entry in sensor_data:
            weights.append(entry.weight)
            bmis.append(entry.bmi)
            ecgis.append(entry.ecg)
            dates.append(entry.datetime.strftime("%Y-%m-%d %H:%M:%S"))

        # Return the data as JSON for use in JavaScript
        return jsonify({
            'dates': dates,
            'weights': weights,
            'bmis': bmis,
            'ecg': ecgis
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"message": "An error occurred while fetching the data."}), 500

@app.route('/plot')
def plot():
    return render_template('sensor_data_plot.html')  
 
@app.route("/dashboard")
@login_required
def dashboard():
    user_data = SensorData.query.filter_by(rfid=current_user.rfid).all()  # ✅ Show only logged-in user's data
    return render_template("dashboard.html", user=current_user, sensor_data=user_data)


@app.route('/api/sensor_data', methods=['GET'])
@login_required
def get_sensor_data():
    try:
        # Fetch all sensor data for the logged-in user
        sensor_data = SensorData.query.filter_by(rfid=current_user.rfid).all()

        # Convert objects to dictionary
        sensor_data_list = []
        for data in sensor_data:
            sensor_data_list.append({
                "id": data.id,
                "rfid": data.rfid,
                "weight": data.weight,
                "height": data.height,
                "bmi": data.bmi,
                "ecg": data.ecg,
                "datetime": data.datetime.strftime("%Y-%m-%d %H:%M:%S")
            })

        return jsonify({"sensor_data": sensor_data_list}), 200

    except Exception as e:
        return jsonify({"message": f"An error occurred: {str(e)}"})
    
if __name__ == '__main__':
    # app.run(debug=True) # Don't use debug=True in production
    app.run(host='0.0.0.0', port=5000, debug=False)
    


















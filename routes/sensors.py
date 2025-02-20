from flask import Blueprint, request, jsonify
from app import db, SensorData

sensors_bp = Blueprint('sensors', __name__)

@sensors_bp.route('/add_sensor_data', methods=['POST'])
def add_sensor_data():
    data = request.json

    try:
        new_entry = SensorData(
            rfid=data['rfid'],
            weight=data['weight'],
            height=data['height'],
            bmi=data['bmi'],
            ecg=data['ecg'],
            datetime=data['datetime'],
            prediction_result=data.get('prediction_result', '')
        )

        db.session.add(new_entry)
        db.session.commit()
        
        return jsonify({"message": "Sensor data added successfully"}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@sensors_bp.route('/get_sensor_data/<rfid>', methods=['GET'])
def get_sensor_data(rfid):
    """Retrieve user sensor data based on RFID"""
    data = SensorData.query.filter_by(rfid=rfid).all()
    
    if not data:
        return jsonify({"error": "No data found"}), 404

    return jsonify([{
        "weight": entry.weight,
        "height": entry.height,
        "bmi": entry.bmi,
        "ecg": entry.ecg,
        "datetime": entry.datetime,
        "prediction_result": entry.prediction_result
    } for entry in data])

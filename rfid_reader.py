import serial
import time
from flask_sqlalchemy import SQLAlchemy
from app import db, User

# Configure the serial port for the RFID reader
SERIAL_PORT = "/dev/ttyUSB0"  # Adjust based on your system
BAUD_RATE = 9600

def read_rfid():
    """Reads RFID card data from the serial port."""
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Allow connection to establish

        while True:
            if ser.in_waiting > 0:
                rfid_tag = ser.readline().decode('utf-8').strip()
                if rfid_tag:
                    print(f"RFID Detected: {rfid_tag}")
                    authenticate_user(rfid_tag)
    
    except serial.SerialException as e:
        print(f"Error: {e}")

def authenticate_user(rfid_tag):
    """Checks if RFID is registered and authenticates user."""
    user = User.query.filter_by(rfid=rfid_tag).first()
    if user:
        print(f"Access Granted: {user.name} ({user.role})")
    else:
        print("Access Denied: Unrecognized RFID")

if __name__ == "__main__":
    read_rfid()

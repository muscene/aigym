from flask import Blueprint, render_template
from flask_login import login_required, current_user
from app import SensorData

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/dashboard')
@login_required
def dashboard():
    """Display user fitness data"""
    data = SensorData.query.filter_by(rfid=current_user.rfid).all()
    
    return render_template('dashboard.html', data=data, user=current_user)

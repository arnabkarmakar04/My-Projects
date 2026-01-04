import os
import io
import json
import zlib
from datetime import datetime, timezone
from PIL import Image
from dotenv import load_dotenv

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import google.generativeai as genai

# Load environment variables
load_dotenv()

app = Flask(__name__)

# 1. CONFIGURATION 

# Security: Random key ensures sessions are invalidated when server restarts
app.secret_key = os.urandom(24) 
app.config['SESSION_PERMANENT'] = False 

# Database Setup (SQLite)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///diagnexus.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Increase max upload size to 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 
db = SQLAlchemy(app)

# Login Manager Setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' 
login_manager.login_message = None 

# Gemini AI Setup
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY not found in .env file.")
    else:
        genai.configure(api_key=api_key)
except Exception as e:
    print(f"Error configuring Gemini API: {e}")

model = genai.GenerativeModel('gemini-2.5-flash')


# 2. DATABASE MODELS 

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    
    # Login Details
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False) 
    
    # Profile Details
    phone = db.Column(db.String(15), nullable=True)
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(10), nullable=True)
    
    # Role Flag
    is_admin = db.Column(db.Boolean, default=False)
    
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    symptom_logs = db.relationship('SymptomLog', backref='user', lazy=True)
    prescriptions = db.relationship('Prescription', backref='user', lazy=True)
    purchases = db.relationship('PurchaseRecord', backref='user', lazy=True)

class SymptomLog(db.Model):
    __tablename__ = 'symptom_logs'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    user_query = db.Column(db.Text, nullable=False)
    
    # Stores parsed diseases for analytics
    probable_diseases = db.Column(db.String(200), nullable=True)
    
    ai_response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

class Prescription(db.Model):
    __tablename__ = 'prescriptions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # File Storage (Compressed)
    filename = db.Column(db.String(200), nullable=False)
    file_data = db.Column(db.LargeBinary, nullable=True) 
    
    # Metadata: Patient Details
    patient_name = db.Column(db.String(100), nullable=True)
    patient_id = db.Column(db.String(50), nullable=True)
    
    # Metadata: Doctor/Hospital Details
    doctor_name = db.Column(db.String(100), nullable=True)
    doctor_contact = db.Column(db.String(100), nullable=True)
    hospital_name = db.Column(db.String(150), nullable=True)
    hospital_address = db.Column(db.String(250), nullable=True)
    hospital_contact = db.Column(db.String(100), nullable=True)
    
    # The Full Report (Markdown for Display)
    extracted_text = db.Column(db.Text, nullable=False)
    
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

class PurchaseRecord(db.Model):
    __tablename__ = 'purchase_records'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    medicine_name = db.Column(db.String(100), nullable=False)
    vendor_name = db.Column(db.String(50), nullable=False)
    
    bill_filename = db.Column(db.String(200), nullable=True)
    extracted_amount = db.Column(db.Float, nullable=True)
    transaction_id = db.Column(db.String(100), nullable=True)
    
    status = db.Column(db.String(20), default='Under Review') 
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

# Create Database Tables
with app.app_context():
    db.create_all()


# 3. AUTHENTICATION ROUTES 

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        login_type = request.form.get('login_type') 
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            # Security Check for Admin Login
            if login_type == 'admin' and not user.is_admin:
                flash('Access Denied: You do not have Admin privileges.')
                return redirect(url_for('login'))
            
            login_user(user)
            return redirect(url_for('admin_dashboard')) if user.is_admin and login_type == 'admin' else redirect(url_for('index'))
        else:
            flash('Invalid username or password')
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('register'))
        
        hashed_pw = generate_password_hash(password, method='scrypt')
        new_user = User(username=username, email=email, password=hashed_pw)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# 4. USER ROUTES 

@app.route('/')
@login_required
def index():
    if current_user.is_admin:
        return redirect(url_for('admin_dashboard'))
    return render_template('index.html', name=current_user.username)

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        current_user.phone = request.form.get('phone')
        current_user.age = request.form.get('age')
        current_user.gender = request.form.get('gender')
        db.session.commit()
        flash('Profile updated successfully!')
        return redirect(url_for('profile'))
        
    return render_template('profile.html', user=current_user)

@app.route('/analyze-symptoms', methods=['POST'])
@login_required
def analyze_symptoms():
    if not request.json or 'prompt' not in request.json:
        return jsonify({"error": "Invalid request."}), 400

    user_prompt = request.json['prompt']

    # MARKDOWN + JSON WRAPPER PROMPT
    system_prompt = f"""
    You are DiagNexus, an expert medical diagnostic AI assistant.
    User's Symptoms: "{user_prompt}"

    TASK:
    1. Analyze symptoms to identify Top 3 probable disease names for database categorization (JSON).
    2. Generate the report for the patient using the EXACT instructions below.

    OUTPUT FORMAT (RAW JSON ONLY):
    {{
      "top_diseases": ["Disease 1", "Disease 2"],
      "markdown_report": "THE FULL MARKDOWN STRING GENERATED FROM INSTRUCTIONS BELOW"
    }}

    INSTRUCTIONS FOR 'markdown_report' CONTENT 
    
    1. **Emergency Check:** If the symptoms indicate a life-threatening emergency (e.g., crushing chest pain, difficulty breathing, stroke signs, severe bleeding), START your response with a bold warning to call emergency services immediately.
    2. **Scope Check:** If the user input is gibberish or not related to health, politely state that you can only assist with medical symptoms.
    3. **Tone:** Be professional, calm, and empathetic.

    ### REQUIRED OUTPUT FORMAT (Markdown):

    1. **Analysis Summary:**
       - A brief 1-sentence summary of what the symptoms suggest.

    2. **Potential Conditions (Top 3 Most Likely):**
       - **[Condition Name]:** Brief explanation of why this fits.
       - *Note: Only list conditions that genuinely match.*

    3. **Detailed Reasoning:**
       - Explain the connection between the specific symptoms provided and the potential conditions.

    4. **Recommended Next Steps:**
       - **Immediate Action:** (e.g., "Go to ER", "Call Doctor", or "Monitor at home")
       - **Home Care:** (Specific advice like hydration, rest, OTC medication)

    5. **Medical Disclaimer:**
       - "I am an AI, not a doctor. This analysis is for informational purposes only. Always consult a healthcare professional."
    """
    
    try:
        response = model.generate_content(system_prompt, generation_config={"response_mime_type": "application/json"})
        data = json.loads(response.text)
        
        # Save disease string for analytics
        diseases_str = ", ".join(data.get('top_diseases', []))
        
        new_log = SymptomLog(
            user_id=current_user.id, 
            user_query=user_prompt, 
            probable_diseases=diseases_str, 
            ai_response=data.get('markdown_report') 
        )
        db.session.add(new_log)
        db.session.commit()
        return jsonify({"result": data.get('markdown_report')})
    except Exception as e:
        print(f"AI Error: {e}")
        return jsonify({"error": "AI Service Unavailable"}), 500

@app.route('/analyze-prescription', methods=['POST'])
@login_required
def analyze_prescription():
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['file']

    try:
        file_bytes = file.read()
        compressed_data = zlib.compress(file_bytes) # Save space
        mime_type = file.mimetype

        # MARKDOWN + JSON WRAPPER PROMPT 
        prompt = """
        You are an expert pharmacist AI helper. 
        TASK 1: Extract metadata for the database (JSON).
        TASK 2: Generate the visual Markdown report using the specific instructions below.

        OUTPUT FORMAT (RAW JSON ONLY):
        {
            "patient_name": "Name or 'Not Mentioned'",
            "patient_id": "ID/MRN or 'N/A'",
            "doctor_name": "Name or 'Unknown'",
            "doctor_contact": "Contact or 'Unknown'",
            "hospital_name": "Name or 'Unknown'",
            "hospital_address": "Address or 'Unknown'",
            "hospital_contact": "Contact or 'Unknown'",
            "markdown_report": "THE FULL MARKDOWN STRING GENERATED FROM INSTRUCTIONS BELOW"
        }

        INSTRUCTIONS FOR 'markdown_report' CONTENT 
        
        1. **Metadata:** Look for Patient Name AND Patient ID (labeled as ID, PID, MRN, UHID, IPD No).
        2. **Differentiation:** Distinguish between the doctor's handwriting (medicines/advice) and the pre-printed hospital footer text.
        3. **Transcription:** Extract Medicine Name, Dosage (e.g., 1-0-1), Duration, and specific Instructions.
        4. **Uncertainty:** If a medicine name is unclear but you are 90% sure based on context, list it, if completely illegible, write "[Illegible]".
        5. **Structure:** YOU MUST use a standard Markdown Table.

        ### REQUIRED OUTPUT FORMAT:
        **Patient Name:** [Name or "Not Mentioned"] | **ID:** [Extract ID/MRN/UHID if visible, else "N/A"] | **Date:** [Date or "Not Mentioned"]

        ### 1. Medicines List
        | Medicine Name | Dosage & Frequency | Duration | Instructions |
        | :| :| :| :|
        | *Name* | *e.g., 1-0-1 (Morning & Night)* | *e.g., 5 Days* | *e.g., After food* |
        
        ### 2. General Advice
        - [List specific doctor's notes (diet/rest). Ignore pre-printed footer text.]

        ### 3. Disclaimer
        *AI-generated transcription. Handwriting can be misinterpreted. Verify with a pharmacist.*
        """

        content_parts = [prompt]
        if mime_type == 'application/pdf':
            content_parts.append({"mime_type": "application/pdf", "data": file_bytes})
        else:
            image_stream = io.BytesIO(file_bytes)
            img = Image.open(image_stream)
            content_parts.append(img)
        
        response = model.generate_content(content_parts, generation_config={"response_mime_type": "application/json"})
        data = json.loads(response.text)
        
        new_pres = Prescription(
            user_id=current_user.id, filename=file.filename, file_data=compressed_data,
            patient_name=data.get('patient_name'), patient_id=data.get('patient_id'),
            doctor_name=data.get('doctor_name'), doctor_contact=data.get('doctor_contact'),
            hospital_name=data.get('hospital_name'), hospital_address=data.get('hospital_address'),
            hospital_contact=data.get('hospital_contact'),
            extracted_text=data.get('markdown_report') 
        )
        db.session.add(new_pres)
        db.session.commit()
        return jsonify({"result": data.get('markdown_report')})
    except Exception as e:
        print(f"File Error: {e}")
        return jsonify({"error": "Failed to analyze document."}), 500

@app.route('/verify-purchase', methods=['POST'])
@login_required
def verify_purchase():
    if 'file' not in request.files: return jsonify({"error": "No bill uploaded"}), 400
    file = request.files['file']
    
    try:
        # Simple placeholder extraction for fraud check
        # In a real app, you would parse the image here to get the 'extracted_amount'
        new_purchase = PurchaseRecord(
            user_id=current_user.id,
            medicine_name=request.form.get('medicine_name', 'Unknown'),
            vendor_name=request.form.get('vendor_name', 'Unknown'),
            bill_filename=file.filename,
            extracted_amount=0.00, # Placeholder until AI runs
            transaction_id="AI_PENDING", 
            status="Under Review"
        )
        db.session.add(new_purchase)
        db.session.commit()
        return jsonify({"result": "Bill uploaded and under review."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 5. ADMIN ROUTES & ANALYTICS 

@app.route('/admin')
@login_required
def admin_dashboard():
    if not current_user.is_admin: 
        return redirect(url_for('index'))
    
    # 1. KPIs
    total_patients = User.query.filter_by(is_admin=False).count()
    total_admins = User.query.filter_by(is_admin=True).count()
    total_revenue = db.session.query(func.sum(PurchaseRecord.extracted_amount)).filter_by(status='Approved').scalar() or 0.0
    pending_count = PurchaseRecord.query.filter_by(status='Under Review').count()
    
    # 2. TOP DISEASES
    all_logs = SymptomLog.query.filter(SymptomLog.probable_diseases != None).all()
    disease_counts = {}
    for log in all_logs:
        if log.probable_diseases:
            for d in log.probable_diseases.split(','):
                d_clean = d.strip()
                if d_clean: disease_counts[d_clean] = disease_counts.get(d_clean, 0) + 1
    top_diseases = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    # 3. TOP MEDICINES
    top_medicines = db.session.query(
        PurchaseRecord.medicine_name, func.count(PurchaseRecord.id)
    ).filter_by(status='Approved').group_by(PurchaseRecord.medicine_name).order_by(func.count(PurchaseRecord.id).desc()).limit(5).all()

    # 4. DEMOGRAPHICS (EXCLUDES ADMINS)
    age_stats = {
        '18-25': User.query.filter(User.is_admin == False, User.age >= 18, User.age <= 25).count(),
        '26-40': User.query.filter(User.is_admin == False, User.age >= 26, User.age <= 40).count(),
        '41-60': User.query.filter(User.is_admin == False, User.age >= 41, User.age <= 60).count(),
        '60+': User.query.filter(User.is_admin == False, User.age > 60).count()
    }
    
    # 5. CORRELATION (Search -> Purchase)
    active_users = db.session.query(User).join(SymptomLog).join(PurchaseRecord).distinct().limit(5).all()

    pending_bills = PurchaseRecord.query.filter_by(status='Under Review').all()
    recent_logs = SymptomLog.query.order_by(SymptomLog.timestamp.desc()).limit(10).all()

    return render_template('admin_dashboard.html', 
                           user=current_user,
                           total_patients=total_patients, total_admins=total_admins,
                           total_revenue=round(total_revenue, 2), pending_count=pending_count,
                           top_diseases=top_diseases, top_medicines=top_medicines,
                           age_stats=age_stats, correlation_users=active_users,
                           pending_bills=pending_bills, recent_logs=recent_logs)

@app.route('/admin/approve/<int:purchase_id>')
@login_required
def approve_purchase(purchase_id):
    if not current_user.is_admin: return redirect(url_for('index'))
    record = db.session.get(PurchaseRecord, purchase_id)
    if record:
        record.status = 'Approved'
        record.extracted_amount = 10.0 # Mock amount
        db.session.commit()
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/reject/<int:purchase_id>')
@login_required
def reject_purchase(purchase_id):
    if not current_user.is_admin: return redirect(url_for('index'))
    record = db.session.get(PurchaseRecord, purchase_id)
    if record:
        record.status = 'Rejected'
        db.session.commit()
    return redirect(url_for('admin_dashboard'))

# USER MANAGEMENT ROUTES 

@app.route('/admin/users')
@login_required
def admin_users():
    if not current_user.is_admin: return redirect(url_for('index'))
    all_users = User.query.order_by(User.created_at.desc()).all()
    return render_template('user_management.html', user=current_user, all_users=all_users)

@app.route('/admin/users/promote/<int:user_id>')
@login_required
def promote_user(user_id):
    if not current_user.is_admin: return redirect(url_for('index'))
    user_to_promote = db.session.get(User, user_id)
    if user_to_promote:
        user_to_promote.is_admin = True
        db.session.commit()
        flash(f"User {user_to_promote.username} is now an Admin.")
    return redirect(url_for('admin_users'))

@app.route('/admin/users/demote/<int:user_id>')
@login_required
def demote_user(user_id):
    if not current_user.is_admin: return redirect(url_for('index'))
    user_to_demote = db.session.get(User, user_id)
    if user_to_demote:
        if user_to_demote.id == current_user.id:
            flash("You cannot demote yourself!")
        else:
            user_to_demote.is_admin = False
            db.session.commit()
            flash(f"User {user_to_demote.username} is now a Patient.")
    return redirect(url_for('admin_users'))

# ANALYTICS ROUTE 

@app.route('/admin/analytics')
@login_required
def admin_analytics():
    if not current_user.is_admin: return redirect(url_for('index'))
    
    # 1. Revenue
    revenue_data = db.session.query(
        func.date(PurchaseRecord.timestamp), 
        func.sum(PurchaseRecord.extracted_amount)
    ).filter(PurchaseRecord.status == 'Approved')\
     .group_by(func.date(PurchaseRecord.timestamp))\
     .order_by(func.date(PurchaseRecord.timestamp)).all()
    
    trend_labels = [r[0] for r in revenue_data]
    trend_values = [r[1] for r in revenue_data]

    # 2. Status
    status_counts = db.session.query(
        PurchaseRecord.status, 
        func.count(PurchaseRecord.id)
    ).group_by(PurchaseRecord.status).all()
    
    status_labels = [s[0] for s in status_counts]
    status_values = [s[1] for s in status_counts]

    # 3. Vendors
    vendor_data = db.session.query(
        PurchaseRecord.vendor_name, 
        func.count(PurchaseRecord.id)
    ).group_by(PurchaseRecord.vendor_name)\
     .order_by(func.count(PurchaseRecord.id).desc()).limit(5).all()
    
    vendor_labels = [v[0] for v in vendor_data]
    vendor_values = [v[1] for v in vendor_data]

    return render_template('analytics.html', 
                           user=current_user,
                           trend_labels=trend_labels, trend_values=trend_values,
                           status_labels=status_labels, status_values=status_values,
                           vendor_labels=vendor_labels, vendor_values=vendor_values)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
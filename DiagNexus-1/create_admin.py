from app import app, db, User
from werkzeug.security import generate_password_hash

with app.app_context():
    # 1. Define your secure admin credentials
    username = "Arnab"
    email = "arnabkarmakar937@gmail.com"
    password = "12345678"

    # 2. Check if admin already exists
    if User.query.filter_by(username=username).first():
        print("Admin user already exists.")
    else:
        # 3. Create the user with is_admin=True
        hashed_pw = generate_password_hash(password, method='scrypt')
        
        # THIS is the only place where is_admin=True is allowed
        new_admin = User(username=username, email=email, password=hashed_pw, is_admin=True)
        
        db.session.add(new_admin)
        db.session.commit()
        print("Success! Admin account created.")
        print(f"Login with: {username} / {password}")
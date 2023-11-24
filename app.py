from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://h0l4dmin:kwkMIlPzfWhDBjFhe2CR@hola-bd-qa.ckeqxfmdqgne.us-east-1.rds.amazonaws.com/incoporacion'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'senku'

# Configuración de la base de datos
db = SQLAlchemy(app)

# Configuración de la extensión de JWT
jwt = JWTManager(app)

# Importar rutas después de inicializar la aplicación y la base de datos
from routes import *

if __name__ == '__main__':
    # Crear las tablas en la base de datos
    with app.app_context():
        db.create_all()

    app.run(debug=True)

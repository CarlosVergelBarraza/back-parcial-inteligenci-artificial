from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from flask_migrate import Migrate
from flask import request, jsonify
from app import app, db
import pandas as pd
from models.models import User, Role
from predecir import preditcModel, train_model


# Ruta para listar todos los usuarios
@app.route('/', methods=['GET'])
def heal():
    return "estamos bien"

#region logeo

# Ruta para registrar un nuevo usuario
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()

    # Verificar que los roles proporcionados existen
    # role_names = data.get('roles', [])
    role_names = ["client"]
    roles = Role.query.filter(Role.name.in_(role_names)).all()
    if len(roles) != len(role_names):
        return jsonify({'message': 'Uno o más roles no válidos'}), 400

    # Intentar eliminar un usuario existente con el mismo nombre
    existing_user = User.query.filter_by(username=data['username']).first()
    if existing_user:
        return jsonify({'message': 'Usuario Ya existe'}), 400
        # db.session.delete(existing_user)
        # db.session.commit()

    # Crear un nuevo usuario y asignar roles
    new_user = User(username=data['username'], password=data['password'])
    new_user.roles.extend(roles)

    # Limpiar la sesión antes de agregar el nuevo usuario
    db.session.expunge_all()

    # Agregar el nuevo usuario a la sesión y realizar la commit
    with app.app_context():
        db.session.add(new_user)
        db.session.commit()

    # Crear un token que incluya información sobre los roles del usuario
    return jsonify({'message': 'Usuario registrado exitosamente!'}), 201

# Ruta para asignar un rol a un usuario existente
@app.route('/assign_role', methods=['POST'])
def assign_role():
    data = request.get_json()

    username = data.get('username')
    role_name = data.get('role')

    # Verificar que el rol proporcionado existe
    role = Role.query.filter_by(name=role_name).first()
    if not role:
        return jsonify({'message': 'Rol no válido'}), 400

    # Verificar que el usuario exista
    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({'message': 'Usuario no encontrado'}), 404

    # Asignar el rol al usuario
    user.roles.append(role)

    # Realizar la commit
    with app.app_context():
        db.session.commit()

    return jsonify({'message': f'Rol {role_name} asignado al usuario {username}'}), 200


# Ruta para autenticación de usuarios
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    with app.app_context():
        user = User.query.filter_by(username=username, password=password).first()

        if user:
            # Crear un token que incluya información sobre los roles del usuario
            roles = [role.name for role in user.roles]
            access_token = create_access_token(identity={'username': user.username, 'roles': roles})
            return jsonify(access_token=access_token, roles=roles)
        else:
            return jsonify({'message': 'Usuario o contraseña incorrectos'}), 401

#endregion

#region listas

# Ruta para listar todos los usuarios
@app.route('/users', methods=['GET'])
def list_users():
    with app.app_context():
        users = User.query.all()
        users_list = []

        for user in users:
            user_info = {
                'id': user.id,
                'username': user.username,
                'roles': [role.name for role in user.roles]
            }
            users_list.append(user_info)

        return jsonify({'users': users_list}), 200
# Ruta para listar roles y usuarios asociados
@app.route('/roles', methods=['GET'])
def list_roles():
    with app.app_context():
        roles = Role.query.all()
        roles_list = []

        for role in roles:
            role_info = {
                'id': role.id,
                'name': role.name,
                'users': [{'id': user.id, 'username': user.username} for user in role.users]
            }
            roles_list.append(role_info)

        return jsonify({'roles': roles_list}), 200
#endregion 


@app.route('/owner_user', methods=['GET'])
@jwt_required()
def owner_user():
    current_user = get_jwt_identity()

    with app.app_context():
        user=current_user['username']
        print(user)
        infouser = User.query.filter_by(username=user).first()
        print(infouser.username)
        return jsonify(identity={'username': infouser.username}), 200
    
@app.route('/update_password', methods=['PUT'])
@jwt_required()
def update_password():
    current_user = get_jwt_identity()

    with app.app_context():
        user=current_user['username']
        print(user)
        infouser = User.query.filter_by(username=user).first()
        print(infouser.username)
        return jsonify(identity={'username': infouser.username}), 200
    
@app.route('/predict', methods=['POST'])
@jwt_required()
def predit():
    try:
        # Receive the form data as JSON
        form_data = request.get_json()
        

        required_fields = ["GENDER", "Car_Owner", "Propert_Owner", "CHILDREN", "Annual_income",
                        "Type_Income", "EDUCATION", "Marital_status", "Housing_type", "Birthday_count", 
                        "Employed_days", "Mobile_phone", "Work_Phone", "Phone", "EMAIL_ID", "Type_Occupation", 
                        "Family_Members"]
        for field in required_fields:
            if form_data.get(field) is None or form_data.get(field) == "":
                return jsonify(erros={'error': f"El campo '{field}' es nulo. Parámetro inválido.", 'field':field}), 200
        current_user = get_jwt_identity()

        with app.app_context():
            user=current_user['username']
            # print(user)
            infouser = User.query.filter_by(username=user).first()
            
            resultado=preditcModel(form_data)
            return jsonify(predict=resultado[0]), 200
    except Exception as e:
        return jsonify({'error': str(e)})
    
#region protegidas
    #region adminRoleOnly
    # Ruta protegida por token JWT y rol de administrador
@app.route('/admin', methods=['GET'])
@jwt_required()
def admin_route():
    current_user = get_jwt_identity()

    with app.app_context():
        if 'admin' in current_user['roles']:
            return jsonify(logged_in_as=current_user), 200
        else:
            return jsonify({'message': 'Acceso denegado. Se requiere rol de administrador'}), 403

# # Ruta para registrar un nuevo usuario con uno o varios roles
# @app.route('/incor/register-adim', methods=['POST'])
# @jwt_required()
# def registerAdmin():
#     current_user = get_jwt_identity()

#     with app.app_context():
#         if 'admin' in current_user['roles']:
#             print("admin create user")
#         else:
#             return jsonify({'message': 'Acceso denegado. Se requiere rol de administrador'}), 403
#     data = request.get_json()

#     # Verificar que los roles proporcionados existen
#     role_names = data.get('roles', [])
#     roles = Role.query.filter(Role.name.in_(role_names)).all()
#     if len(roles) != len(role_names):
#         return jsonify({'message': 'Uno o más roles no válidos'}), 400

#     # Intentar eliminar un usuario existente con el mismo nombre
#     existing_user = User.query.filter_by(username=data['username']).first()
#     if existing_user:
#         db.session.delete(existing_user)
#         db.session.commit()

#     # Crear un nuevo usuario y asignar roles
#     new_user = User(username=data['username'], password=data['password'])
#     new_user.roles.extend(roles)

#     # Limpiar la sesión antes de agregar el nuevo usuario
#     db.session.expunge_all()

#     # Agregar el nuevo usuario a la sesión y realizar la commit
#     with app.app_context():
#         db.session.add(new_user)
#         db.session.commit()

#     return jsonify({'message': 'Usuario registrado exitosamente!'}), 201
   #endregion

# Ruta para crear un nuevo rol
@app.route('/create_role', methods=['POST'])
@jwt_required()
def create_role():
    data = request.get_json()

    new_role = Role(name=data['name'])

    with app.app_context():
        db.session.add(new_role)
        db.session.commit()

    return jsonify({'message': 'Rol creado exitosamente!'}), 201

#endregion
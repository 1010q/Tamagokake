from fastapi import Depends, FastAPI, HTTPException, status, Header
from functools import wraps
from passlib.context import CryptContext
from pydantic import BaseModel
from supabase import create_client, Client
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import joblib

app = FastAPI()
SUPABASE_URL = "https://vsmlnrzidfzdmvawficj.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZzbWxucnppZGZ6ZG12YXdmaWNqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzE5ODk0MjQsImV4cCI6MjA0NzU2NTQyNH0.NFBj-NAG0i_MwV7pV7dbmxwIuifsvwfF3Br9-_Nsw0A"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


class User(BaseModel):
    username: str

class SignInRequest(BaseModel):
    email: str
    password: str

class SignUpRequest(BaseModel):
    email: str
    password: str
    username: str



def authenticate_user(func):
    @wraps(func)
    async def wrapper(*args, authorization: str = Header(...), **kwargs):
        try:
            token = authorization.split(" ")[1]
            user_response = supabase.auth.get_user(token)
            username = user_response.user.user_metadata.get('username')
            user_id = user_response.user.id
            user_info = {"user_id": user_id, "username": username}
        except Exception as e:
            raise HTTPException(status_code=401, detail="Invalid or missing token") from e
        # user_info をエンドポイントに渡す
        return await func(*args, user_info=user_info, **kwargs)
    return wrapper


@app.post("/login")
async def login_user(request: SignInRequest):
    auth_response = supabase.auth.sign_in_with_password({"email": request.email,"password": request.password,})
    if "error" in auth_response:
        raise HTTPException(status_code=400, detail=auth_response.error.message)

    return {"access_token": auth_response.session.access_token, "token_type": "bearer"}


@app.post("/register")
async def register_user(request: SignUpRequest):
    supabase.auth.sign_up({
    'email': request.email,
    'password': request.password,
    "options": {"data": {"username": request.username}},
    })
    sign_in_response = supabase.auth.sign_in_with_password({"email": request.email, "password": request.password})
    access_token = sign_in_response.session.access_token
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/logout")
async def logout_user(authorization: str = Header(...)):
    token = authorization.split(" ")[1]
    auth_response = supabase.auth.sign_out(access_token=token)
    if "error" in auth_response:
        raise HTTPException(status_code=400, detail=auth_response.error.message)

    return {"message": "ログアウト成功"}


@app.get("/")
async def home(user_info: dict):
    user_id = user_info.get('user_id')


@app.post("/material/input/allmodel")
@authenticate_user
async def get_model(user_info: dict):
    user_id = user_info.get('user_id')
    X = np.array([[180, 72, 4],[200, 66, 5],[180, 72, 6],[240, 66, 5],[180, 60, 8],[180, 72, 4]])  # 特徴量: ご飯の量, 卵の量, 評価
    y = np.array([5.8, 7.0, 7.0, 7.1, 8.0, 5.0])

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)


@app.post("/material/input/eachmodel")
@authenticate_user
async def get_model(user_info: dict, rice_amount: int, egg_amount: int):
    user_id = user_info.get('user_id')
    response = supabase.table("materials").select("*").filter("user_id", "eq", user_id).execute()
    data = response.get("data") 
    data_count = len(response.get("data"))

    default_X = np.array([[180, 72, 4],[200, 66, 5],[180, 72, 6],[240, 66, 5],[180, 60, 8],[180, 72, 4]])
    default_y = np.array([5.8, 7.0, 7.0, 7.1, 8.0, 5.0])

    if data_count < 5: # データが少なければ、デフォルトデータと重み付きユーザーデータを使用
        user_X = np.array([[d["rice_amount"], d["egg_amount"], d["rating"]] for d in data])
        user_y = np.array([d["soysauce_amount"] for d in data])

        weight = 3
        user_X_weighted = np.repeat(user_X, weight, axis=0)
        user_y_weighted = np.repeat(user_y, weight, axis=0)

        X = np.vstack((default_X, user_X_weighted))
        y = np.concatenate((default_y, user_y_weighted))

    else: # データが十分にあればユーザーデータのみ使用
        X = np.array([[d["rice_amount"], d["egg_amount"], d["rating"]] for d in data])
        y = np.array([d["soysauce_amount"] for d in data])

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    new_data = np.array([[rice_amount, egg_amount, 5]])  # ご飯と卵の量、評価(5固定)
    new_data_poly = poly.transform(new_data)
    predicted_soy_sauce = model.predict(new_data_poly)
    predicted_soy_sauce = round(predicted_soy_sauce,1)
    return {"predicted_soy_sauce": predicted_soy_sauce}
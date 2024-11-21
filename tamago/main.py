from fastapi import Depends, FastAPI, UploadFile, Form, File
from pydantic import BaseModel
from supabase import create_client, Client
from fastapi.security import OAuth2PasswordBearer
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from uuid import uuid4

app = FastAPI()
SUPABASE_URL = "https://vsmlnrzidfzdmvawficj.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZzbWxucnppZGZ6ZG12YXdmaWNqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzE5ODk0MjQsImV4cCI6MjA0NzU2NTQyNH0.NFBj-NAG0i_MwV7pV7dbmxwIuifsvwfF3Br9-_Nsw0A"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class Signin(BaseModel):
    email: str
    password: str

class Signup(BaseModel):
    email: str
    password: str
    username: str

class MaterialInput(BaseModel):
    rice_amount: int
    egg_amount: int

class Rating(BaseModel):
    soysauce_amount: float
    rice_amount: int
    egg_amount: int
    rating: int
    model: str=None


@app.post("/login")
async def login_user(request: Signin):
    auth_response = supabase.auth.sign_in_with_password({"email": request.email,"password": request.password,})

    return {"access_token": auth_response.session.access_token, "token_type": "bearer"}


@app.post("/register")
async def register_user(request: Signup):
    sign_up_response = supabase.auth.sign_up({'email': request.email, 'password': request.password})
    sign_in_response = supabase.auth.sign_in_with_password({"email": request.email, "password": request.password})
    user_id = sign_up_response.user.id

    supabase.table("profile").insert({
        "user_id": user_id,
        "username": request.username,
        "image_url": None  # 初期値はNone
    }).execute()
    access_token = sign_in_response.session.access_token
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/logout")
async def logout_user(token: str = Depends(oauth2_scheme)):
    supabase.auth.sign_out(access_token=token)

    return {"message": "ログアウト成功"}


@app.post("/material/input/allmodel")
async def get_model( request: MaterialInput, token: str = Depends(oauth2_scheme)):
    user_response = supabase.auth.get_user(token)
    current_user_id = user_response.user.id

    response = supabase.table("allmodel").select("*").execute()
    data = response.data

    X = np.array([[d["rice_amount"], d["egg_amount"], d["rating"]] for d in data])
    y = np.array([d["soysauce_amount"] for d in data])

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    new_data = np.array([[request.rice_amount, request.egg_amount, 5]])  # ご飯と卵の量、評価(5固定)
    new_data_poly = poly.transform(new_data)
    predicted_soy_sauce = model.predict(new_data_poly)
    soysauce_amount = round(predicted_soy_sauce[0], 1)

    return {"predicted_soy_sauce": soysauce_amount, "rice_amount": request.rice_amount, "egg_amount": request.egg_amount, "model": "allmodel"}


@app.post("/material/input/eachmodel")
async def get_model(request: MaterialInput, token: str = Depends(oauth2_scheme)):
    user_response = supabase.auth.get_user(token)
    user_id = user_response.user.id
    try:
        response = supabase.table("eachmodel").select("*").filter("user_id", "eq", user_id).execute()
        data = response.data
        data_count = len(data)
    except Exception as e:
        data_count = 0

    X = np.array([[180, 72, 4],[200, 66, 5],[180, 72, 6],[240, 66, 5],[180, 60, 8],[180, 72, 4]])
    y = np.array([5.8, 7.0, 7.0, 7.1, 8.0, 5.0])
    if data_count == 0:
        pass

    elif data_count < 5: # データが少なければ、デフォルトデータと重み付きユーザーデータを使用
        user_X = np.array([[d["rice_amount"], d["egg_amount"], d["rating"]] for d in data])
        user_y = np.array([d["soysauce_amount"] for d in data])

        weight = 3
        user_X_weighted = np.repeat(user_X, weight, axis=0)
        user_y_weighted = np.repeat(user_y, weight, axis=0)

        X = np.vstack((X, user_X_weighted))
        y = np.concatenate((y, user_y_weighted))

    else: # データが十分にあればユーザーデータのみ使用
        sorted_data = sorted(data, key=lambda d: d["timestamp"], reverse=True) # 新しいデータ3件を重みづけ
        recent_data = sorted_data[:3]

        X = np.array([[d["rice_amount"], d["egg_amount"], d["rating"]] for d in data])
        y = np.array([d["soysauce_amount"] for d in data])

        recent_X = np.array([[d["rice_amount"], d["egg_amount"], d["rating"]] for d in recent_data])
        recent_y = np.array([d["soysauce_amount"] for d in recent_data])
        weight = 3
        recent_X_weighted = np.repeat(recent_X, weight, axis=0)
        recent_y_weighted = np.repeat(recent_y, weight, axis=0)
        X = np.vstack((X, recent_X_weighted))
        y = np.concatenate((y, recent_y_weighted))

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    new_data = np.array([[request.rice_amount, request.egg_amount, 5]])  # ご飯と卵の量、評価(5固定)
    new_data_poly = poly.transform(new_data)
    predicted_soy_sauce = model.predict(new_data_poly)
    soysauce_amount = round(predicted_soy_sauce[0], 1)

    return {"predicted_soy_sauce": soysauce_amount, "rice_amount": request.rice_amount, "egg_amount": request.egg_amount, "model": "eachmodel"}


@app.post("/user/TKG/rating")
async def rating(request: Rating, token: str = Depends(oauth2_scheme)):
    user_response = supabase.auth.get_user(token)
    current_user_id = user_response.user.id

    if request.model == "eachmodel":
       response = (supabase.table("eachmodel").insert({
            "user_id": current_user_id,
            "rice_amount": request.rice_amount,
            "egg_amount": request.egg_amount,
            "soysauce_amount": request.soysauce_amount,
            "rating": request.rating,
        }).execute())
        
    else:
        response = (supabase.table("allmodel").insert({
            "rice_amount": request.rice_amount,
            "egg_amount": request.egg_amount,
            "soysauce_amount": request.soysauce_amount,
            "rating": request.rating,
        }).execute())
        
    return {"message": "評価を保存しました"}


@app.post("/add/eachmodel")
async def add_eachmodel(request: Rating, token: str = Depends(oauth2_scheme)):
    user_response = supabase.auth.get_user(token)
    user_id = user_response.user.id
    response = (supabase.table("eachmodel").insert({
            "user_id": user_id,
            "rice_amount": request.rice_amount,
            "egg_amount": request.egg_amount,
            "soysauce_amount": request.soysauce_amount,
            "rating": request.rating,
        }).execute())

    return {"message": "評価を保存しました"}


@app.get("/")
async def home(token: str = Depends(oauth2_scheme)):
    user_response = supabase.auth.get_user(token)
    current_user_id = user_response.user.id
    user_data = supabase.table("profile").select("username", "image_url").eq("user_id", current_user_id).execute()
    username = user_data.data[0]["username"]
    image_url = user_data.data[0]["image_url"]

    response = supabase.table("notifications").select("*").eq("user_id", current_user_id).execute()
    notifications = response.data

    posts_response = supabase.table("post").select("*").order("created_at", desc=True).execute()
    posts = posts_response.data

    return {
            "user_id": current_user_id, 
            "username": username, 
            "user_image_url": image_url, 
            "posts": posts, 
            "notifications": notifications
        }


@app.patch("/update_posts_sort") # 投稿一覧をスターの多い順にソート
async def update_posts_sort(token: str = Depends(oauth2_scheme)):
    user_response = supabase.auth.get_user(token)
    current_user_id = user_response.user.id

    posts_response = supabase.table("post").select("*").order("star_count", desc=True).execute()
    posts = posts_response.data

    return {"posts": posts}


@app.get("/profile/{user_id}")
async def get_profile(user_id: str, token: str = Depends(oauth2_scheme)):
    user_response = supabase.auth.get_user(token)
    current_user_id = user_response.user.id

    user_data = supabase.table("profile").select("username", "image_url").eq("user_id", current_user_id).execute()
    username = user_data.data[0]["username"]
    image_url = user_data.data[0]["image_url"]

    posts_responce = supabase.table("post").select("*").eq("user_id", user_id).execute()
    posts = posts_responce.data

    total_stars = sum(post["stars"] for post in posts) if posts else 0

    starred_posts_response = supabase.table("post").select("id").eq("user_id", user_id).eq("stars", 1).execute()
    starred_post_ids = [item["id"] for item in starred_posts_response.data] if starred_posts_response.data else []

    return {"posts": posts, "username": username, "user_image_url": image_url, "star_count": total_stars, "starred_posts": starred_post_ids}


@app.post("/profile/update/{user_id}")
async def update_profile(user_id: str, username: str=Form(None), user_image: UploadFile=Form(None), token: str = Depends(oauth2_scheme)):
    user_response = supabase.auth.get_user(token)
    current_user_id = user_response.user.id
    updated_data = {}
    if username:
        updated_data["username"] = username

    if user_image:
        image_path = f"profile_images/{uuid4()}"
        image_content = await user_image.read()
        response = supabase.storage.from_("post_image").upload(image_path, image_content, {"content-type": "image/png"})
        image_url = f"{SUPABASE_URL}/storage/v1/object/public/post_image/{image_path}"
        updated_data["image_url"] = image_url

    if updated_data:
        supabase.table("profile").update(updated_data).eq("user_id", user_id).execute()

    return {"message": "プロフィールを更新しました", "updated_data": updated_data, "image_url": image_url}


@app.post("/post/create/")
async def create_post(title: str = Form(...), description: str = Form(...), image: UploadFile = File(...), token: str = Depends(oauth2_scheme)):
    user_response = supabase.auth.get_user(token)
    user_id = user_response.user.id
    image_path = f"posts/{uuid4()}"
    file_content = await image.read()
    response = supabase.storage.from_("post_image").upload(image_path, file_content, {"content-type": "image/png"})
    image_url = f"{SUPABASE_URL}/storage/v1/object/public/post_image/{image_path}"

    response = (supabase.table("post").insert({
        "user_id": user_id,
        "title": title,
        "description": description,
        "image_url": image_url
    }).execute())

    post_id = response.data[0]["id"]

    return {"post_id": post_id}


@app.get("/postdetail/{post_id}")
async def get_post_detail(post_id: str, token: str = Depends(oauth2_scheme)):
    user_response = supabase.auth.get_user(token)
    current_user_id = user_response.user.id

    response = supabase.table("post").select("*").eq("post_id", post_id).execute()
    data = response.data

    post_detail = data[0]
    comment_response = supabase.table("comments").select("*").eq("post_id", post_id).execute()
    comments = comment_response.data

    return {"post": post_detail, "comments": comments}


@app.post("/postdetail/{post_id}/comment")
async def add_comment(post_id: str, content: str, token: str = Depends(oauth2_scheme)):
    user_response = supabase.auth.get_user(token)
    user_id = user_response.user.id

    comment_data = {
        "post_id": post_id,
        "user_id": user_id,
        "content": content,
    }
    response = supabase.table("comments").insert(comment_data).execute()

    post_response = supabase.table("post").select("user_id").eq("post_id", post_id).execute()
    author_id = post_response.data[0]["author_id"]

    notification_message = "あなたの投稿にコメントがつきました"
    notification_data = {
        "user_id": author_id,
        "type": notification_message,
        "post_id": post_id
    }
    supabase.table("notifications").insert(notification_data).execute()


    return {"message": "コメントを追加しました"}


@app.patch("/project/{post_id}/star")
async def toggle_star(post_id: str, token: str = Depends(oauth2_scheme)):
    user_response = supabase.auth.get_user(token)
    current_user_id = user_response.user.id

    star_entry = supabase.table("stars").select("id").eq("user_id", current_user_id).eq("post_id", post_id).execute()

    post_response = supabase.table("post").select("star_count").eq("post_id", post_id).execute()
    post = post_response.data[0]
    star_count = post["star_count"]

    if star_entry.data:
        supabase.table("stars").delete().eq("user_id", current_user_id).eq("post_id", post_id).execute() # すでにスターを押している場合は削除
        star_count -= 1
    else:
        supabase.table("stars").insert({"user_id": current_user_id, "post_id": post_id, "starred": True}).execute() # スターを押していない場合は追加
        star_count += 1

    supabase.table("post").update({"star_count": star_count}).eq("post_id", post_id).execute()

    return {"star_count": star_count}
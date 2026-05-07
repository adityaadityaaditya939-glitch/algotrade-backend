from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from jose import JWTError, jwt
from datetime import datetime, timedelta
import psycopg2, psycopg2.extras, os, bcrypt
from dotenv import load_dotenv

load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET", "algotrade-super-secret-change-in-production")
JWT_ALGO   = "HS256"
JWT_EXPIRE = int(os.getenv("JWT_EXPIRE_MINUTES", 1440))

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "dbname":   os.getenv("DB_NAME", "algotrade"),
    "user":     os.getenv("DB_USER", "algouser"),
    "password": os.getenv("DB_PASS", "yourpassword"),
}

bearer        = HTTPBearer()
router        = APIRouter(prefix="/api/auth",   tags=["auth"])
orders_router = APIRouter(prefix="/api/orders", tags=["orders"])

def get_conn():
    return psycopg2.connect(**DB_CONFIG)

def init_db():
    sql = """
    CREATE TABLE IF NOT EXISTS users (
        id         SERIAL PRIMARY KEY,
        email      VARCHAR(255) UNIQUE NOT NULL,
        username   VARCHAR(100) NOT NULL,
        hashed_pw  TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT NOW(),
        last_login TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS trade_orders (
        id         SERIAL PRIMARY KEY,
        user_id    INTEGER REFERENCES users(id) ON DELETE CASCADE,
        pair       VARCHAR(20)   NOT NULL,
        side       VARCHAR(10)   NOT NULL,
        order_type VARCHAR(20)   DEFAULT 'market',
        amount     NUMERIC(18,8) NOT NULL,
        price      NUMERIC(18,8),
        leverage   INTEGER       DEFAULT 1,
        status     VARCHAR(20)   DEFAULT 'paper',
        created_at TIMESTAMP     DEFAULT NOW()
    );
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        print("✅ Database tables initialized")
    except Exception as e:
        conn.rollback()
        print(f"❌ DB init error: {e}")
    finally:
        conn.close()

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8")[:72], bcrypt.gensalt(rounds=12)).decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8")[:72], hashed.encode("utf-8"))

def create_token(user_id: int, email: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE)
    return jwt.encode({"sub": str(user_id), "email": email, "exp": expire}, JWT_SECRET, algorithm=JWT_ALGO)

def get_current_user(creds: HTTPAuthorizationCredentials = Depends(bearer)):
    try:
        payload = jwt.decode(creds.credentials, JWT_SECRET, algorithms=[JWT_ALGO])
        return {"user_id": int(payload["sub"]), "email": payload["email"]}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

class RegisterRequest(BaseModel):
    email:    EmailStr
    username: str
    password: str

class LoginRequest(BaseModel):
    email:    EmailStr
    password: str

class OrderRequest(BaseModel):
    pair:       str
    side:       str
    order_type: str   = "market"
    amount:     float
    price:      float | None = None
    leverage:   int   = 1

@router.post("/register")
def register(body: RegisterRequest):
    if len(body.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id FROM users WHERE email = %s", (body.email.lower(),))
            if cur.fetchone():
                raise HTTPException(400, "Email already registered")
            hashed = hash_password(body.password)
            cur.execute(
                "INSERT INTO users (email, username, hashed_pw) VALUES (%s,%s,%s) RETURNING id",
                (body.email.lower(), body.username, hashed)
            )
            user_id = cur.fetchone()["id"]
        conn.commit()
        return {
            "message": "Account created successfully",
            "token":   create_token(user_id, body.email.lower()),
            "user":    {"id": user_id, "email": body.email.lower(), "username": body.username}
        }
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(500, f"Registration error: {str(e)}")
    finally:
        conn.close()

@router.post("/login")
def login(body: LoginRequest):
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, email, username, hashed_pw FROM users WHERE email = %s",
                (body.email.lower(),)
            )
            user = cur.fetchone()
            # ✅ Proper password verification
            if not user or not verify_password(body.password, user["hashed_pw"]):
                raise HTTPException(401, "Invalid email or password")
            cur.execute("UPDATE users SET last_login = NOW() WHERE id = %s", (user["id"],))
        conn.commit()
        return {
            "message": "Login successful",
            "token":   create_token(user["id"], user["email"]),
            "user":    {"id": user["id"], "email": user["email"], "username": user["username"]}
        }
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(500, f"Login error: {str(e)}")
    finally:
        conn.close()

@router.get("/me")
def me(current_user: dict = Depends(get_current_user)):
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id, email, username, created_at, last_login FROM users WHERE id = %s",
                       (current_user["user_id"],))
            user = cur.fetchone()
            if not user: raise HTTPException(404, "User not found")
            return {k: str(v) if v else None for k, v in dict(user).items()}
    finally:
        conn.close()

@router.post("/logout")
def logout(current_user: dict = Depends(get_current_user)):
    return {"message": "Logged out successfully"}

@orders_router.post("/")
def place_order(body: OrderRequest, current_user: dict = Depends(get_current_user)):
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "INSERT INTO trade_orders (user_id,pair,side,order_type,amount,price,leverage) VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (current_user["user_id"], body.pair, body.side, body.order_type, body.amount, body.price, body.leverage)
            )
            row = cur.fetchone()
        conn.commit()
        return {"message": "Order placed", "order_id": row["id"]}
    except Exception as e:
        conn.rollback()
        raise HTTPException(500, str(e))
    finally:
        conn.close()

@orders_router.get("/")
def get_orders(current_user: dict = Depends(get_current_user)):
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM trade_orders WHERE user_id=%s ORDER BY created_at DESC LIMIT 50",
                       (current_user["user_id"],))
            return [dict(o) for o in cur.fetchall()]
    finally:
        conn.close()

# Run auth server:  python auth.py
# Or combined:      add include_router(router) to backend.py

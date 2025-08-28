import os
import io
import re
import json
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict
import numpy as np
import pandas as pd
import hashlib
import streamlit as st

# ---------------------------
# Database paths
# ---------------------------
DB_DIR = "/tmp/taxintellilytics"
USERS_DB = f"{DB_DIR}/users.db"
HISTORY_DB = f"{DB_DIR}/taxintellilytics_history.sqlite"

os.makedirs(DB_DIR, exist_ok=True)

# ---------------------------
# Initialize Users DB
# ---------------------------
def init_user_db():
    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT,
            salt TEXT,
            subscription_expiry TEXT,
            plan TEXT
        );
    ''')
    conn.commit()
    conn.close()

init_user_db()

# ---------------------------
# Initialize History DB
# ---------------------------
def init_history_db():
    conn = sqlite3.connect(HISTORY_DB)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS income_tax_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_name TEXT,
            taxpayer_type TEXT,
            year INTEGER,
            period TEXT,
            revenue REAL,
            cogs REAL,
            opex REAL,
            other_income REAL,
            other_expenses REAL,
            pbit REAL,
            capital_allowances REAL,
            exemptions REAL,
            taxable_income REAL,
            gross_tax REAL,
            credits_wht REAL,
            credits_foreign REAL,
            rebates REAL,
            net_tax_payable REAL,
            metadata_json TEXT,
            created_at TEXT
        );
    ''')
    conn.commit()
    conn.close()

init_history_db()

# ---------------------------
# Hash helpers
# ---------------------------
def rand_salt():
    return hashlib.sha256(os.urandom(16)).hexdigest()[:16]

def hash_with_salt(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

# ---------------------------
# User DB helpers
# ---------------------------
def add_user_to_db(username, password_hash, salt, expiry=None, plan=None):
    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO users (username, password_hash, salt, subscription_expiry, plan) VALUES (?,?,?,?,?)",
        (username, password_hash, salt, expiry, plan)
    )
    conn.commit()
    conn.close()

def get_user_record(username):
    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    c.execute(
        "SELECT username, password_hash, salt, subscription_expiry, plan FROM users WHERE username=?",
        (username,)
    )
    row = c.fetchone()
    conn.close()
    return row

def check_subscription(username):
    rec = get_user_record(username)
    if rec and rec[3]:
        expiry = datetime.strptime(rec[3], "%Y-%m-%d")
        return expiry >= datetime.now()
    return False

def update_subscription(username, days, plan):
    expiry = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    c.execute(
        "UPDATE users SET subscription_expiry=?, plan=? WHERE username=?",
        (expiry, plan, username)
    )
    conn.commit()
    conn.close()

# ---------------------------
# Subscription plans
# ---------------------------
SUBSCRIPTION_PLANS = {
    "Basic - 500,000 UGX/month": {"amount": 500_000, "days": 30},
    "Standard - 1,000,000 UGX/month": {"amount": 1_000_000, "days": 30},
    "Premium - 1,500,000 UGX/month": {"amount": 1_500_000, "days": 30},
    "Annual - 10% of monthly × 12": {"amount": 500_000*12*0.90, "days": 365}  # 10% discount
}

# ---------------------------
# Streamlit page setup
# ---------------------------
st.set_page_config(page_title="TaxIntellilytics — Income Tax (Uganda)", layout="wide")
st.title("💼 TaxIntellilytics")

# ---------------------------
# Session state defaults
# ---------------------------
for key, val in [
    ("authenticated", False),
    ("current_user", None),
    ("subscription_active", False),
    ("plan", None),
    ("payment_pending", False),
    ("pending_plan", None)
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------------------
# Helper stubs for missing functions
# ---------------------------
def create_payment_link(username, amount):
    # Simulate payment link creation
    return f"https://pay.example.com/{username}?amount={amount}"

def check_subscription_db(username):
    return check_subscription(username)

def update_subscription_db(username, days, plan):
    update_subscription(username, days, plan)

def show_tax_module():
    st.info("Main tax module goes here. (Stub)")

def parse_financial_file_bytes(file_bytes, filename):
    if filename.endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    elif filename.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(file_bytes))
    else:
        raise ValueError("Unsupported file type")

@st.cache_data
def auto_map_pl_cached(df_json):
    # Dummy auto-mapping logic
    df = pd.read_json(df_json)
    revenue = float(df[df.columns[-1]].sum())  # Just a placeholder
    return revenue, 0.0, 0.0, 0.0, 0.0

def save_history(row):
    conn = sqlite3.connect(HISTORY_DB)
    c = conn.cursor()
    keys = ','.join(row.keys())
    qmarks = ','.join(['?']*len(row))
    c.execute(f"INSERT INTO income_tax_history ({keys}) VALUES ({qmarks})", tuple(row.values()))
    conn.commit()
    conn.close()

def validate_and_build_return(form_code, payload):
    # Validate required fields
    for k, v in payload.items():
        if v is None or (isinstance(v, str) and not v.strip()):
            raise ValueError(f"Missing value for {k}")
    # Build DataFrame
    df = pd.DataFrame([payload])
    return df

# ---------------------------
# Tabs: Login / Sign Up
# ---------------------------
tab_login, tab_signup = st.tabs(["Login / Renew", "Sign Up"])

# ---------------------------
# LOGIN / RENEW
# ---------------------------
with tab_login:
    st.subheader("🔑 Login / Renew Subscription")
    login_username = st.text_input("Username", key="login_user_tab")
    login_password = st.text_input("Password", type="password", key="login_pass_tab")

    if st.button("Login / Renew", key="login_btn_tab"):
        rec = get_user_record(login_username)
        if not rec:
            st.error("Unknown user")
        else:
            stored_hash, salt, expiry, plan = rec[1], rec[2], rec[3], rec[4]
            if hash_with_salt(login_password, salt) == stored_hash:
                st.session_state["authenticated"] = True
                st.session_state["current_user"] = login_username
                st.session_state["subscription_active"] = check_subscription_db(login_username)
                st.session_state["plan"] = plan
                st.success(f"Welcome {login_username} 👋")

                if not st.session_state["subscription_active"]:
                    st.warning("🚨 Your subscription is inactive. Choose a plan to activate below.")
            else:
                st.error("Incorrect password")

    # Subscription renewal
    if st.session_state.get("authenticated") and not st.session_state["subscription_active"]:
        selected_plan = st.selectbox(
            "Select a subscription plan", 
            list(SUBSCRIPTION_PLANS.keys()), 
            key="login_plan_tab"
        )
        if st.button("Subscribe via MTN/Airtel", key="pay_btn_tab"):
            plan_info = SUBSCRIPTION_PLANS[selected_plan]
            pay_link = create_payment_link(st.session_state["current_user"], amount=plan_info["amount"])
            if pay_link:
                st.markdown(f"[Click here to pay via MTN/Airtel]({pay_link})")
                st.session_state["payment_pending"] = True
                st.session_state["pending_plan"] = selected_plan

# ---------------------------
# SIGN UP
# ---------------------------
with tab_signup:
    st.subheader("📝 Sign Up & Subscribe")
    signup_username = st.text_input("Choose a username", key="signup_user_tab")
    signup_password = st.text_input("Choose a password", type="password", key="signup_pass_tab")
    signup_plan = st.selectbox(
        "Choose a subscription plan", 
        list(SUBSCRIPTION_PLANS.keys()), 
        key="signup_plan_tab"
    )

    if st.button("Sign Up & Subscribe", key="signup_btn_tab"):
        if get_user_record(signup_username):
            st.error("Username already exists")
        else:
            salt = rand_salt()
            hashed = hash_with_salt(signup_password, salt)
            add_user_to_db(signup_username, hashed, salt, expiry=None, plan=signup_plan)
            st.success("Account created! Proceed to payment.")
            plan_info = SUBSCRIPTION_PLANS[signup_plan]
            pay_link = create_payment_link(signup_username, amount=plan_info["amount"])
            if pay_link:
                st.markdown(f"[Click here to pay via MTN/Airtel]({pay_link})")
                st.session_state["payment_pending"] = True
                st.session_state["pending_plan"] = signup_plan

# ---------------------------
# POST-PAYMENT CONFIRMATION
# ---------------------------
if st.session_state.get("authenticated") and st.session_state.get("payment_pending"):
    st.info("Waiting for payment confirmation...")
    if st.button("Confirm Payment (Demo)"):
        plan_info = SUBSCRIPTION_PLANS[st.session_state["pending_plan"]]
        update_subscription_db(st.session_state["current_user"], days=plan_info["days"], plan=st.session_state["pending_plan"])
        st.session_state["subscription_active"] = True
        st.session_state["payment_pending"] = False
        st.success(f"✅ Subscription activated for {st.session_state['pending_plan']}!")
        show_tax_module()

# ---------------------------
# LOGGED-IN VIEW
# ---------------------------
if st.session_state["authenticated"] and st.session_state["subscription_active"]:
    st.sidebar.write(f"Welcome {st.session_state['current_user']} 👋")
    if st.button("Logout", key="logout_btn_tab"):
        st.session_state["authenticated"] = False
        st.session_state["current_user"] = None
        st.session_state["subscription_active"] = False
        st.session_state["plan"] = None
        st.session_state["payment_pending"] = False
        st.session_state["pending_plan"] = None
    show_tax_module()

# ---------------------------
# The rest of your tax computation, audit, and dashboard code goes here...
# (You can copy your original logic for tabs, forms, and calculations below this point)
# ---------------------------

# ----------------------------
# Footer / App Info
# ----------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size:12px; color:gray;'>
        TaxIntellilytics &copy; 2025 | Developed by Walter Hillary Kaijamahe
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Authentication Logout
# ----------------------------
if st.session_state.get("authenticated"):
    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        st.experimental_rerun()

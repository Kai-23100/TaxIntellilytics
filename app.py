import os
import io
import re
import json
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import hashlib
import streamlit as st
import requests

# ---------------------------
# Config: writable DB paths for Streamlit Cloud
# ---------------------------
DB_DIR = os.path.join(os.getenv("STREAMLIT_TMP_DIR", "/tmp"), "taxintellilytics")
os.makedirs(DB_DIR, exist_ok=True)
USERS_DB = os.path.join(DB_DIR, "users.db")
HISTORY_DB = os.path.join(DB_DIR, "taxintellilytics_history.sqlite")

# ---------------------------
# FLUTTERWAVE PAYMENT (placeholder keys)
# ---------------------------
FLUTTERWAVE_SECRET_KEY = os.getenv("FLW_SECRET", "FLWSECK_TEST-xxxxxxxxxxxx")
FLUTTERWAVE_PUBLIC_KEY = os.getenv("FLW_PUBLIC", "FLWPUBK_TEST-xxxxxxxxxxxx")

def create_payment_link(username, amount=5000, currency="UGX"):
    url = "https://api.flutterwave.com/v3/payments"
    headers = {
        "Authorization": f"Bearer {FLUTTERWAVE_SECRET_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "tx_ref": f"sub_{username}_{datetime.now().timestamp()}",
        "amount": amount,
        "currency": currency,
        "redirect_url": "http://localhost:8501",
        "customer": {"email": f"{username}@example.com", "name": username},
        "customizations": {"title": "TaxIntellilytics Subscription", "description": "Access premium tax analytics"}
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json().get("data", {}).get("link")
        st.error("Payment initialization failed.")
    except Exception:
        st.error("Payment initialization failed (network error).")
    return None

# ---------------------------
# USER DB helpers
# ---------------------------

def init_user_db():
    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT,
            salt TEXT,
            subscription_expiry TEXT
        );
    ''')
    conn.commit()
    conn.close()


def add_user_to_db(username: str, password_hash: str, salt: str, expiry: str = None):
    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO users (username, password_hash, salt, subscription_expiry) VALUES (?,?,?,?)",
              (username, password_hash, salt, expiry))
    conn.commit()
    conn.close()


def get_user_record(username: str):
    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    c.execute("SELECT username, password_hash, salt, subscription_expiry FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    return row


def update_subscription_db(username: str, days: int = 30):
    expiry = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    c.execute("UPDATE users SET subscription_expiry=? WHERE username=?", (expiry, username))
    conn.commit()
    conn.close()


def check_subscription_db(username: str) -> bool:
    rec = get_user_record(username)
    if rec and rec[3]:
        try:
            expiry = datetime.strptime(rec[3], "%Y-%m-%d")
            return expiry >= datetime.now()
        except Exception:
            return False
    return False

# ---------------------------
# Hash helpers
# ---------------------------

def rand_salt():
    # deterministic salt for demo; replace with os.urandom for production
    return hashlib.sha256(b"TaxIntelliSalt").hexdigest()[:16]


def hash_with_salt(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

# ---------------------------
# TAX HISTORY DB helpers
# ---------------------------

def init_tax_db():
    with sqlite3.connect(HISTORY_DB) as conn:
        conn.execute("""
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
        """)


def save_history(row: dict):
    with sqlite3.connect(HISTORY_DB) as conn:
        cols = ",".join(row.keys())
        placeholders = ",".join(["?"] * len(row))
        conn.execute(f"INSERT INTO income_tax_history ({cols}) VALUES ({placeholders})", list(row.values()))
        conn.commit()


@st.cache_data(ttl=600)
def load_history_cached(client_filter: str = "") -> pd.DataFrame:
    with sqlite3.connect(HISTORY_DB) as conn:
        df = pd.read_sql_query("SELECT * FROM income_tax_history ORDER BY year DESC, created_at DESC", conn)
    if client_filter and not df.empty:
        df = df[df["client_name"].str.contains(client_filter, case=False, na=False)]
    return df

# ---------------------------
# Initialize DBs and seed demo users only if empty
# ---------------------------
init_user_db()
init_tax_db()

conn = sqlite3.connect(USERS_DB)
c = conn.cursor()
c.execute("SELECT COUNT(*) FROM users")
count = c.fetchone()[0]
conn.close()

if count == 0:
    _demo_users = [
        ("user1", "12345", True),
        ("user2", "password", False),
    ]
    for uname, pwd, has_sub in _demo_users:
        salt = rand_salt()
        phash = hash_with_salt(pwd, salt)
        expiry = (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d") if has_sub else None
        add_user_to_db(uname, phash, salt, expiry)

# ---------------------------
# Tax computation & other utilities (kept as in your original file)
# ---------------------------

def compute_individual_tax_brackets(taxable_income: float, brackets: List[Dict]) -> float:
    if taxable_income <= 0:
        return 0.0
    tax = 0.0
    for i, b in enumerate(brackets):
        threshold = b["threshold"]
        rate = b["rate"]
        fixed = b.get("fixed", 0.0)
        next_threshold = brackets[i + 1]["threshold"] if i + 1 < len(brackets) else None

        if taxable_income > threshold:
            upper = taxable_income if next_threshold is None else min(taxable_income, next_threshold)
            taxable_slice = max(0.0, upper - threshold)
            tax = fixed + taxable_slice * rate
        else:
            break
    return round(max(0.0, tax), 2)


def compute_company_tax(taxable_income: float, company_rate: float = 0.30) -> float:
    if taxable_income <= 0:
        return 0.0
    return round(taxable_income * company_rate, 2)


def apply_credits_and_rebates(gross_tax: float, credits_wht: float, credits_foreign: float, rebates: float) -> float:
    return max(0.0, gross_tax - credits_wht - credits_foreign - rebates)

# (keep URA_SCHEMAS, harmonize_tb, audit_findings, etc. unchanged)

URA_SCHEMAS = {
    "DT-2001": {
        "title": "Income Tax Return Form for Individual with Business Income",
        "fields": [
            ("TIN", "str"),
            ("Taxpayer Name", "str"),
            ("Period", "str"),
            ("Year", "int"),
            ("Business Income (UGX)", "float"),
            ("Allowable Deductions (UGX)", "float"),
            ("Capital Allowances (UGX)", "float"),
            ("Exemptions (UGX)", "float"),
            ("Taxable Income (UGX)", "float"),
            ("Gross Tax (UGX)", "float"),
            ("WHT Credits (UGX)", "float"),
            ("Foreign Tax Credit (UGX)", "float"),
            ("Rebates (UGX)", "float"),
            ("Net Tax Payable (UGX)", "float"),
        ],
    },
    # ... other schemas kept as-is (DT-2002, DT-2003, DT-2004)
}

DEFAULT_CONTROL_MAP = pd.DataFrame([
    ["Cash & Bank", "Debit", r"(?i)cash|bank|current account|cash at hand", 50_000],
    ["Accounts Receivable", "Debit", r"(?i)ar|trade receivable|debtors", 50_000],
    ["Inventory", "Debit", r"(?i)inventory|stock", 50_000],
    ["Accounts Payable", "Credit", r"(?i)ap|trade payable|creditors", 50_000],
    ["VAT Payable", "Credit", r"(?i)vat|output vat|vat payable", 50_000],
    ["VAT Receivable", "Debit", r"(?i)input vat|vat receivable", 50_000],
    ["PAYE Payable", "Credit", r"(?i)paye|pay as you earn", 50_000],
    ["WHT Receivable", "Debit", r"(?i)withholding tax (receivable|asset)|wht receivable|wht asset", 50_000],
    ["Income Tax Payable", "Credit", r"(?i)income tax payable|corporation tax payable", 50_000],
    ["Share Capital", "Credit", r"(?i)share capital|stated capital", 50_000],
    ["Retained Earnings", "Credit", r"(?i)retained earnings|accumulated (profit|loss)", 50_000],
], columns=["Category", "NormalBalance", "Patterns", "MaterialityUGX"])

# (keep harmonize_tb, re_sum, match_control_amounts, pnl_totals_from_tb, audit_findings unchanged)

def harmonize_tb(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    df2 = df.copy()
    for wanted in ["account", "description"]:
        if wanted in cols and "account" not in df2.columns:
            df2.rename(columns={cols[wanted]: "Account"}, inplace=True)

    debit_col = None
    credit_col = None
    for c in df2.columns:
        cl = c.lower()
        if "debit" in cl and debit_col is None:
            debit_col = c
        if "credit" in cl and credit_col is None:
            credit_col = c

    if debit_col and credit_col:
        df2["Debit"] = pd.to_numeric(df2[debit_col], errors="coerce").fillna(0.0)
        df2["Credit"] = pd.to_numeric(df2[credit_col], errors="coerce").fillna(0.0)
        df2["Amount"] = df2["Debit"] - df2["Credit"]
    else:
        amt_col = None
        for c in df2.columns:
            if c.lower() in ["amount", "balance", "closing balance", "ending balance", "net"]:
                amt_col = c
                break
        if amt_col is None:
            num_cols = df2.select_dtypes(include="number").columns
            if len(num_cols) > 0:
                amt_col = num_cols[-1]
        if amt_col is None:
            raise ValueError("Could not detect Amount/Debit/Credit columns in Trial Balance.")
        df2["Amount"] = pd.to_numeric(df2[amt_col], errors="coerce").fillna(0.0)
        df2["Debit"] = df2["Amount"].clip(lower=0)
        df2["Credit"] = (-df2["Amount"]).clip(lower=0)

    if "Account" not in df2.columns:
        df2["Account"] = df2.iloc[:, 0].astype(str)

    return df2[["Account", "Debit", "Credit", "Amount"]]


def re_sum(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").fillna(0.0).sum())


def match_control_amounts(tb_df: pd.DataFrame, control_map: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, row in control_map.iterrows():
        cat = row["Category"]
        pat = str(row["Patterns"]).strip()
        nb = row["NormalBalance"]
        materiality = float(row["MaterialityUGX"])
        if not pat:
            amt = 0.0
        else:
            mask = tb_df["Account"].astype(str).str.contains(pat, regex=True, na=False)
            amt = re_sum(tb_df.loc[mask, "Amount"])
        expected_sign = 1 if nb.lower().startswith("debit") else -1
        signed_ok = (amt == 0) or (np.sign(amt) == expected_sign)
        exception = 0.0
        if not signed_ok:
            exception = abs(amt)
        elif abs(amt) < materiality:
            exception = 0.0
        else:
            exception = 0.0
        out.append({
            "Category": cat,
            "NormalBalance": nb,
            "MatchedAmount": amt,
            "ExpectedSign": "Debit" if expected_sign == 1 else "Credit",
            "SignOK": signed_ok,
            "MaterialityUGX": materiality,
            "ExceptionUGX": exception
        })
    return pd.DataFrame(out)


def pnl_totals_from_tb(tb_df: pd.DataFrame) -> Dict[str, float]:
    acc = tb_df["Account"].astype(str)
    is_income = acc.str.contains(r"(?i)(income|revenue|sales|gain)")
    is_cogs = acc.str.contains(r"(?i)cogs|cost of goods|cost of sales")
    is_opex = acc.str.contains(r"(?i)expense|utilities|rent|salary|transport|admin|repairs|maintenance")
    is_other_income = acc.str.contains(r"(?i)other income|finance income|interest income|dividend income|gain")
    is_other_exp = acc.str.contains(r"(?i)other expense|finance cost|interest expense|loss")

    amt = tb_df["Amount"].astype(float)

    revenue = -re_sum(amt[is_income])
    cogs = re_sum(amt[is_cogs])
    opex = re_sum(amt[is_opex])
    other_income = -re_sum(amt[is_other_income])
    other_expenses = re_sum(amt[is_other_exp])

    return dict(revenue=revenue, cogs=cogs, opex=opex, other_income=other_income, other_expenses=other_expenses)


def audit_findings(tb_df: pd.DataFrame,
                   control_map: pd.DataFrame,
                   mapped_pl: Dict[str, float],
                   materiality_total_ugx: float = 100_000.0) -> Dict[str, pd.DataFrame]:
    debit_total = re_sum(tb_df["Debit"])
    credit_total = re_sum(tb_df["Credit"])
    tb_diff = round(debit_total - credit_total, 2)

    tb_integrity = pd.DataFrame([{
        "TotalDebit": debit_total,
        "TotalCredit": credit_total,
        "Difference(should be 0)": tb_diff,
        "Pass": abs(tb_diff) < 1e-2
    }])

    ctrl = match_control_amounts(tb_df, control_map)

    tb_pl = pnl_totals_from_tb(tb_df)
    pl_comp = pd.DataFrame([
        {"Item": "Revenue", "TB": tb_pl["revenue"], "Mapped": mapped_pl.get("revenue", 0.0)},
        {"Item": "COGS", "TB": tb_pl["cogs"], "Mapped": mapped_pl.get("cogs", 0.0)},
        {"Item": "OPEX", "TB": tb_pl["opex"], "Mapped": mapped_pl.get("opex", 0.0)},
        {"Item": "Other Income", "TB": tb_pl["other_income"], "Mapped": mapped_pl.get("other_income", 0.0)},
        {"Item": "Other Expenses", "TB": tb_pl["other_expenses"], "Mapped": mapped_pl.get("other_expenses", 0.0)},
    ])
    pl_comp["Delta"] = pl_comp["Mapped"] - pl_comp["TB"]
    pl_comp["Material"] = pl_comp["Delta"].abs() >= materiality_total_ugx

    exceptions = []
    if not tb_integrity["Pass"].iloc[0]:
        exceptions.append({"Area": "TB Integrity", "Issue": "Debits != Credits", "Amount": tb_diff})

    for _, r in ctrl.iterrows():
        if not bool(r["SignOK"]):
            exceptions.append({
                "Area": f"Control: {r['Category']}",
                "Issue": f"Sign mismatch (Expected {r['ExpectedSign']})",
                "Amount": float(r["MatchedAmount"])
            })
        elif abs(float(r["MatchedAmount"])) >= float(r["MaterialityUGX"]):
            pass

    for _, r in pl_comp.iterrows():
        if bool(r["Material"]):
            exceptions.append({
                "Area": f"P&L Reconciliation — {r['Item']}",
                "Issue": "Delta exceeds materiality",
                "Amount": float(r["Delta"])
            })

    exceptions_df = pd.DataFrame(exceptions) if exceptions else pd.DataFrame(columns=["Area", "Issue", "Amount"])
    notable_ctrl = ctrl.loc[ctrl["MatchedAmount"].abs() >= ctrl["MaterialityUGX"], ["Category", "MatchedAmount", "ExpectedSign"]].copy()

    return {
        "tb_integrity": tb_integrity,
        "control_accounts": ctrl,
        "pl_comparison": pl_comp,
        "exceptions": exceptions_df,
        "notable_controls": notable_ctrl
    }

# ---------------------------
# Authentication flow (try authenticator lib, fall back to DB)
# ---------------------------

st.set_page_config(page_title="TaxIntellilytics — Income Tax (Uganda)", layout="wide")
st.title("💼 TaxIntellilytics — Income Tax (Uganda)")

USE_AUTH_LIB = False
authenticator = None

# demo credentials (used only for building authenticator payload if lib available)
_demo_usernames = ["user1", "user2"]
_demo_passwords = ["12345", "password"]

try:
    import streamlit_authenticator as stauth
    try:
        # Try the modern API: Hasher(list_of_passwords).generate()
        hashed_passwords = stauth.Hasher(_demo_passwords).generate()
        user_dict = {u: {"name": u, "password": hp} for u, hp in zip(_demo_usernames, hashed_passwords)}
        authenticator = stauth.Authenticate(
            {"usernames": user_dict},
            cookie_name="taxintellilytics_cookie",
            key="taxintellilytics_signature",
            cookie_expiry_days=1
        )
        USE_AUTH_LIB = True
    except TypeError:
        # Incompatible Hasher signature in this environment
        USE_AUTH_LIB = False
except Exception:
    USE_AUTH_LIB = False

username, name, auth_status = None, None, None

if USE_AUTH_LIB and authenticator is not None:
    # Use the library's login widget
    name, auth_status, username = authenticator.login("Login", "sidebar")
    if auth_status:
        st.sidebar.write(f"Welcome {name} 👋")
        authenticator.logout("Logout", "sidebar")

        if check_subscription_db(username):
            st.success("✅ Subscription active! Access granted.")
            show_tax_module()
        else:
            st.warning("🚨 You need an active subscription to access TaxIntellilytics.")
            if st.button("Subscribe with Flutterwave"):
                link = create_payment_link(username)
                if link:
                    st.markdown(f"[Click here to pay via Flutterwave]({link})")
    else:
        if auth_status is False:
            st.error("Username/password is incorrect")
        else:
            st.warning("Please enter your username and password")
else:
    # Fallback built-in login form
    st.info("Using built-in login (fallback). This ensures the app runs even if the authenticator library is incompatible.")
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
        st.session_state["current_user"] = None

    if not st.session_state["authenticated"]:
        with st.form("login_form", clear_on_submit=False):
            username_input = st.text_input("Username")
            password_input = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
        if submitted:
            rec = get_user_record(username_input)
            if not rec:
                st.error("Unknown user")
            else:
                stored_hash = rec[1]
                salt = rec[2] or rand_salt()
                if hash_with_salt(password_input, salt) == stored_hash:
                    st.session_state["authenticated"] = True
                    st.session_state["current_user"] = username_input
                    st.success(f"Welcome {username_input} 👋")
                    st.experimental_rerun()
                else:
                    st.error("Incorrect password")
    else:
        cur_user = st.session_state["current_user"]
        st.sidebar.write(f"Welcome {cur_user} 👋")
        if st.button("Logout"):
            st.session_state["authenticated"] = False
            st.session_state["current_user"] = None
            st.experimental_rerun()

        if check_subscription_db(cur_user):
            st.success("✅ Subscription active! Access granted.")
            show_tax_module()
        else:
            st.warning("🚨 You need an active subscription to access TaxIntellilytics.")
            if st.button("Simulate Subscribe (demo)"):
                update_subscription_db(cur_user, days=365)
                st.experimental_rerun()

              
# ----------------------------
# Sidebar configuration
# ----------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    taxpayer_type = st.selectbox("Taxpayer Type", ["company", "individual"], key="sb_taxpayer_type")
    tax_year = st.number_input("Year", min_value=2000, max_value=datetime.now().year, value=datetime.now().year, step=1, key="sb_tax_year")
    period_label = st.text_input("Period label (e.g., FY2024/25)", value=f"FY{tax_year}", key="sb_period_label")

    st.markdown("### Individual Progressive Brackets (editable JSON)")
    default_brackets = [
        {"threshold": 0.0, "rate": 0.0, "fixed": 0.0},
        {"threshold": 2_820_000.0, "rate": 0.10, "fixed": 0.0},
        {"threshold": 4_020_000.0, "rate": 0.20, "fixed": 120_000.0},
        {"threshold": 4_920_000.0, "rate": 0.30, "fixed": 360_000.0},
        {"threshold": 10_000_000.0, "rate": 0.40, "fixed": 1_830_000.0},
    ]
    brackets_json = st.text_area("Brackets JSON", value=json.dumps(default_brackets, indent=2), height=180, key="sb_brackets_json")
    try:
        individual_brackets = sorted(json.loads(brackets_json), key=lambda x: x["threshold"])
    except Exception:
        individual_brackets = default_brackets

    st.markdown("### Company Rate")
    company_rate = st.number_input("Company Income Tax Rate", min_value=0.0, max_value=1.0, value=0.30, step=0.01, key="sb_company_rate")

    st.markdown("### Audit Materiality (UGX)")
    audit_materiality = st.number_input("P&L Reconciliation Materiality", min_value=0.0, value=100_000.0, step=10_000.0, key="sb_audit_mat")

    st.markdown("### Control Account Map (editable)")
    if "control_map" not in st.session_state:
        st.session_state["control_map"] = DEFAULT_CONTROL_MAP.copy()
    control_map = st.data_editor(
        st.session_state["control_map"],
        key="sb_control_map_editor",
        use_container_width=True,
        num_rows="dynamic"
    )
    st.session_state["control_map"] = control_map

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1) Data Import", "2) P&L Mapping", "3) Compute & Credits", "4) Dashboard", "5) Export & URA Forms", "6) Audit & Controls"
])

# Initialize session state placeholders
if "pl_df" not in st.session_state:
    st.session_state["pl_df"] = None
if "mapped_values" not in st.session_state:
    st.session_state["mapped_values"] = {}
if "addbacks_values" not in st.session_state:
    st.session_state["addbacks_values"] = {}
if "allowables_values" not in st.session_state:
    st.session_state["allowables_values"] = {}

# ----------------------------
# Tab 1: Data Import
# ----------------------------
with tab1:
    st.subheader("📂 Upload Financials (CSV/XLSX) or Connect to QuickBooks (Optional)")

    # Simulated QuickBooks connection button
    def qb_connect_button():
        import pandas as pd
        # Temporary placeholder until actual connection logic is implemented
        return pd.DataFrame()  # empty DataFrame as placeholder

    # Call the QuickBooks connect button (simulated)
    qb_df = qb_connect_button()

    # File uploader
    uploaded = st.file_uploader(
        "Upload P&L / Trial Balance (CSV or Excel)",
        type=["csv", "xlsx"],
        help="Upload a Profit & Loss or Trial Balance export from your accounting system",
        key="t1_file_uploader"
    )

    df = None

    # Prefer QuickBooks data if available
    if qb_df is not None and not qb_df.empty:
        df = qb_df

    # Parse uploaded file if provided
    if uploaded is not None:
        uploaded_bytes = uploaded.getvalue()
        try:
            df = parse_financial_file_bytes(uploaded_bytes, uploaded.name)
        except Exception as e:
            st.error(f"Failed to parse file: {e}")
            df = None

    # Display data preview
    if df is not None and not df.empty:
        st.session_state["pl_df"] = df
        st.write("### Preview (first 50 rows)")
        st.dataframe(df.head(50), use_container_width=True)

        if st.button("Auto-Map P&L (fast)", key="t1_btn_automap"):
            df_json = df.head(10000).to_json()
            revenue, cogs, opex, other_income, other_expenses = auto_map_pl_cached(df_json)
            st.session_state["mapped_values"] = {
                "revenue": revenue,
                "cogs": cogs,
                "opex": opex,
                "other_income": other_income,
                "other_expenses": other_expenses
            }
            st.success("Auto-map completed — check P&L Mapping tab to adjust/confirm.")
    else:
        st.info("Upload a file or use QuickBooks (simulated) to proceed.")

# ----------------------------
# Tab 2: P&L Mapping
# ----------------------------
with tab2:
    st.subheader("🧭 Map P&L → Revenue / COGS / OPEX / Other")
    if st.session_state["pl_df"] is None:
        st.warning("No data found. Go to 'Data Import' first or manually enter P&L below.")
    else:
        df = st.session_state["pl_df"].copy()
        st.write("Auto-detect common columns (Account/Amount) or provide manual values.")
        if st.button("Auto-Map (cached)", key="t2_btn_automap"):
            df_json = df.head(10000).to_json()
            revenue, cogs, opex, other_income, other_expenses = auto_map_pl_cached(df_json)
            st.session_state["mapped_values"] = {
                "revenue": revenue, "cogs": cogs, "opex": opex,
                "other_income": other_income, "other_expenses": other_expenses
            }
            st.success("Auto-mapping complete (you can override the values).")

    mv = st.session_state.get("mapped_values", {})
    st.markdown("#### Manual / Override entries (edit and press Update)")
    with st.form("t2_form_pnl_manual", clear_on_submit=False):
        revenue = st.number_input("Revenue (UGX)", min_value=0.0, value=float(mv.get("revenue", 0.0)), step=1000.0, format="%.2f", key="t2_in_revenue")
        cogs = st.number_input("COGS (UGX)", min_value=0.0, value=float(mv.get("cogs", 0.0)), step=1000.0, format="%.2f", key="t2_in_cogs")
        opex = st.number_input("Operating Expenses (UGX)", min_value=0.0, value=float(mv.get("opex", 0.0)), step=1000.0, format="%.2f", key="t2_in_opex")
        other_income = st.number_input("Other Income (UGX)", min_value=0.0, value=float(mv.get("other_income", 0.0)), step=1000.0, format="%.2f", key="t2_in_oi")
        other_expenses = st.number_input("Other Expenses (UGX)", min_value=0.0, value=float(mv.get("other_expenses", 0.0)), step=1000.0, format="%.2f", key="t2_in_oe")
        update_btn = st.form_submit_button("Update P&L Mapping", use_container_width=True)
    if update_btn:
        st.session_state["mapped_values"] = {
            "revenue": revenue, "cogs": cogs, "opex": opex,
            "other_income": other_income, "other_expenses": other_expenses
        }
        st.success("P&L mapping updated.")

    mv = st.session_state.get("mapped_values", {})
    pbit_manual = (mv.get("revenue", 0.0) + mv.get("other_income", 0.0)) - (mv.get("cogs", 0.0) + mv.get("opex", 0.0) + mv.get("other_expenses", 0.0))
    st.metric("Derived Profit / (Loss) Before Allowances (PBIT)", f"UGX {pbit_manual:,.2f}")

# ----------------------------
# Tab 3: Compute & Credits
# ----------------------------
with tab3:
    st.subheader("🧮 Compute Tax, Apply Credits & Exemptions")

    client_name = st.text_input("Client Name", value="Acme Ltd", key="t3_client_name")
    tin = st.text_input("TIN (optional)", key="t3_tin")

    mv = st.session_state.get("mapped_values", {})
    revenue = mv.get("revenue", 0.0)
    cogs = mv.get("cogs", 0.0)
    opex = mv.get("opex", 0.0)
    other_income = mv.get("other_income", 0.0)
    other_expenses = mv.get("other_expenses", 0.0)
    pbit = (revenue + other_income) - (cogs + opex + other_expenses)

    st.markdown("### P&L summary (derived)")
    st.dataframe(pd.DataFrame([{
        "Revenue": revenue, "COGS": cogs, "OPEX": opex,
        "Other Income": other_income, "Other Expenses": other_expenses,
        "PBIT": pbit
    }]).T, use_container_width=True)

    # --- Addbacks (Disallowables)
    addbacks_labels = [
        "Depreciation (Section 22(3)(b))","Amortisation","Redundancy",
        "Domestic/Private Expenditure (Section 22(3)(a))",
        "Capital Gain (Sections 22(1)(b), 47, 48)","Rental Income Loss (Section 22(1)(c))",
        "Expenses Exceeding 50% of Rental Income (Section 22(2))",
        "Capital Nature Expenditure (Section 22(3)(b))","Recoverable Expenditure (Section 22(3)(c))",
        "Income Tax Paid Abroad (Section 22(3)(d))","Capitalised Income (Section 22(3)(e))",
        "Gift Cost not in Recipient Income (Section 22(3)(f))","Fines or Penalties (Section 22(3)(g))",
        "Employee Retirement Contributions (Section 22(3)(h))","Life Insurance Premiums (Section 22(3)(i))",
        "Pension Payments (Section 22(3)(j))","Alimony / Allowance (Section 22(3)(k))",
        "Suppliers without TIN > UGX5M (Section 22(3)(l))","EFRIS Suppliers w/o e-invoices (Section 22(3)(m))",
        "Debt Obligation Principal (Section 25)","Interest on Capital Assets (Sections 22(3) & 50(2))",
        "Interest on Fixed Capital (Section 25(1))","Bad Debts Recovered (Section 61)",
        "General Provision for Bad Debts (Section 24)","Entertainment Income (Section 23)",
        "Meal & Refreshment Expenses (Section 23)","Charitable Donations to Non-Exempt Orgs (Section 33(1))",
        "Charitable Donations >5% Chargeable Income (Section 33(3))","Legal Fees",
        "Legal Expenses - Capital Items (Section 50)","Legal Expenses - New Trade Rights",
        "Legal Expenses - Breach of Law","Cost of Breach of Contract - Capital Account",
        "Legal Expenses on Breach of Contract - Capital Account","Legal Expenses on Loan Renewals - Non-commercial",
        "Bad Debts by Senior Employee/Management","General Provisions Bad Debts (FI Credit Classification)",
        "Loss on Sale of Fixed Assets (Section 22(3)(b))","Loss on Other Capital Items (Section 22(3)(b))",
        "Expenditure on Share Capital Increase (Section 22(3)(b))","Dividends Paid (Section 22(3)(d))",
        "Provision for Bad Debts (Non-Financial Institutions) (Section 24)",
        "Increase in Provision for Bad Debts (Section 24)","Debt Collection Expenses related to Capital Expenditure",
        "Foreign Currency Debt Gains (Section 46(2))","Costs incidental to Capital Asset (Stamp Duty, Section 50)",
        "Non-Business Expenses (Section 22)","Miscellaneous Staff Costs",
        "Staff Costs - Commuting (Section 22(4)(b))","First Time Work Permits",
        "Unrealised Foreign Exchange Losses (Section 46(3))","Foreign Currency Debt Losses (Section 46)",
        "Education Expenditure (Non Section 32)","Donations (Non Section 33)",
        "Decommissioning Expenditure by Licensee (Section 99(2))","Telephone Costs (10%)",
        "Revaluation Loss","Interest Expense on Treasury Bills (Section 139(e))",
        "Burial Expenses (Section 22(3)(b))","Subscription (Section 22(3)(a))",
        "Interest on Directors Debit Balances (Section 22(3)(a))","Entertainment Expenses (Section 23)",
        "Gifts (Section 22(3)(f))","Dividends Paid (duplicate)","Income Carried to Reserve Fund (Section 22(3)(e))",
        "Impairment Losses on Loans and Advances","Interest Expense on Treasury Bonds (Section 139(e))",
        "Staff Leave Provisions (Section 22(4)(b))","Increase in Gratuity","Balancing Charge (Sections 27(5) & 18(1))"
    ]

    with st.expander("Addbacks (Disallowable Expenses) — click to edit and save", expanded=False):
        with st.form("t3_form_addbacks"):
            ab_values = {}
            for label in addbacks_labels:
                key = f"t3_ab_{re.sub(r'[^a-z0-9]+', '_', label.lower())}"
                default_val = float(st.session_state["addbacks_values"].get(key, 0.0))
                ab_values[key] = st.number_input(label, min_value=0.0, value=default_val, format="%.2f", key=key + "_widget")
            addbacks_submit = st.form_submit_button("Save Addbacks", use_container_width=True)
            if addbacks_submit:
                st.session_state["addbacks_values"].update(ab_values)
                st.success("Addbacks saved to session.")

    total_addbacks = sum(float(v) for v in st.session_state["addbacks_values"].values())
    adjusted_profit = pbit + total_addbacks
    st.markdown(f"### Adjusted Profit (PBIT + Addbacks): UGX {adjusted_profit:,.2f}")

    # --- Allowables (Deductions)
    allowables_labels = [
        "Wear & Tear (Section 27(1))","Industrial Building Allowance (5% for 20 years) (Section 28(1))",
        "Startup Costs (25%) (Section 28)","Reverse VAT (Section 22(1)(a))",
        "Listing Business with Uganda Stock Exchange (Section 29(2)(a))",
        "Registration Fees, Accountant Fees, Legal Fees, Advertising, Training (Section 29(2)(b))",
        "Expenses in Acquiring Intangible Asset (Section 30(1))","Disposal of Intangible Asset (Section 30(2))",
        "Minor Capital Expenditure (Minor Capex) (Section 26(2))","Revenue Expenditures - Repairs & Maintenance (Section 26)",
        "Expenditure on Scientific Research (Section 31(1))","Expenditure on Training (Education) (Section 32(1))",
        "Charitable Donations to Exempt Organisations (Section 33(1))","Charitable Donations Up to 5% Chargeable Income (Section 33(3))",
        "Expenditure on Farming (Section 34)","Apportionment of Deductions (Section 35)",
        "Carry Forward Losses from Previous Period (Section 36(1))","Carry Forward Losses Upto 50% after 7 Years (Section 36(6))",
        "Disposal of Trading Stock (Section 44(1))","Foreign Currency Debt Loss (Realised Exchange Loss) (Section 46(3))",
        "Loss on Disposal of Asset (Section 48)","Exclusion of Doctrine Mutuality (Section 59(3))",
        "Partnership Loss for Resident Partner (Section 66(3))","Partnership Loss for Non-Resident Partner (Section 66(4))",
        "Expenditure or Loss by Trustee Beneficiary (Section 71(5))","Expenditure or Loss by Beneficiary of Deceased Estate (Section 72(2))",
        "Limitation on Deduction for Petroleum Operations (Section 91(1))","Decommission Costs & Expenditures - Petroleum (Section 99(2))",
        "Unrealised Gains (Section 46)","Impairment of Asset","Decrease in Provision for Bad Debts (Section 24)",
        "Bad Debts Written Off (Section 24)","Staff Costs - Business Travel (Section 22)",
        "Private Employer Disability Tax (Section 22(1)(e))","Rental Income Expenditure & Losses (Section 22(1)(c)(2))",
        "Local Service Tax (Section 22(1)(d))","Interest Income on Treasury Bills (Section 139(a))",
        "Interest on Circulating Capital","Interest Income on Treasury Bonds (Section 139(a))",
        "Specific Provisions for Bad Debts (Financial Institutions)","Revaluation Gains (Financial Institutions)",
        "Rental Income (Section 5(3)(a))","Interest Income from Treasury Bills (Section 139(a)(c)(d))",
        "Interest Income from Treasury Bonds (Section 139(a)(c)(d))","Legal Expenses on Breach of Contract to Revenue Account",
        "Legal Expenses on Maintenance of Capital Assets","Legal Expenses on Existing Trade Rights",
        "Legal Expenses Incidental to Revenue Items","Legal Expenses on Debt Collection - Trade Debts",
        "Closing Tax Written Down Value < UGX1M (Section 27(6))","Intangible Assets",
        "Legal Expenses for Renewal of Loans (Financial Institutions)","Interest on Debt Obligation (Loan) (Section 25(1))",
        "Interest on Debt Obligation by Group Member (30% EBITDA) (Section 25(3))","Gains & Losses on Disposal of Assets (Section 22(1)(b))",
        "Balancing Allowance (Sections 27(7))"
    ]

    with st.expander("Allowables (Deductions) — click to edit and save", expanded=False):
        with st.form("t3_form_allowables"):
            al_values = {}
            for label in allowables_labels:
                key = f"t3_al_{re.sub(r'[^a-z0-9]+', '_', label.lower())}"
                default_val = float(st.session_state["allowables_values"].get(key, 0.0))
                al_values[key] = st.number_input(label, min_value=0.0, value=default_val, format="%.2f", key=key + "_widget")
            allowables_submit = st.form_submit_button("Save Allowables", use_container_width=True)
            if allowables_submit:
                st.session_state["allowables_values"].update(al_values)
                st.success("Allowables saved to session.")

    total_allowables = sum(float(v) for v in st.session_state["allowables_values"].values())
    chargeable_income = max(0.0, adjusted_profit - total_allowables)
    st.markdown(f"### Chargeable Income (after allowables): UGX {chargeable_income:,.2f}")

    st.markdown("### Credits, Capital Allowances & Rebates")
    col1, col2, col3 = st.columns(3)
    with col1:
        capital_allowances = st.number_input("Capital Allowances (UGX)", min_value=0.0, value=0.0, format="%.2f", key="t3_in_capital_allowances")
        exemptions = st.number_input("Exemptions (UGX)", min_value=0.0, value=0.0, format="%.2f", key="t3_in_exemptions")
    with col2:
        credits_wht = st.number_input("WHT Credits (UGX)", min_value=0.0, value=0.0, format="%.2f", key="t3_in_wht")
        credits_foreign = st.number_input("Foreign Tax Credit (UGX)", min_value=0.0, value=0.0, format="%.2f", key="t3_in_ftc")
    with col3:
        rebates = st.number_input("Rebates (UGX)", min_value=0.0, value=0.0, format="%.2f", key="t3_in_rebates")
        provisional_tax_paid = st.number_input("Provisional Tax Paid (UGX)", min_value=0.0, value=0.0, format="%.2f", key="t3_in_provisional")

    if st.button("Compute Tax Liability", key="t3_btn_compute"):
        adjusted_taxable_income = max(0.0, chargeable_income - capital_allowances - exemptions)
        if taxpayer_type.lower() == "company":
            gross_tax = compute_company_tax(adjusted_taxable_income, company_rate=company_rate)
        else:
            gross_tax = compute_individual_tax_brackets(adjusted_taxable_income, individual_brackets)

        net_tax_payable = apply_credits_and_rebates(gross_tax, credits_wht, credits_foreign, rebates)
        net_tax_after_provisional = max(0.0, net_tax_payable - provisional_tax_paid)

        st.session_state["last_computation"] = {
            "client_name": client_name,
            "TIN": tin,
            "taxpayer_type": taxpayer_type,
            "year": int(tax_year),
            "period": period_label,
            "revenue": revenue, "cogs": cogs, "opex": opex,
            "other_income": other_income, "other_expenses": other_expenses,
            "pbit": pbit, "total_addbacks": total_addbacks, "total_allowables": total_allowables,
            "capital_allowances": capital_allowances, "exemptions": exemptions,
            "taxable_income": adjusted_taxable_income, "gross_tax": gross_tax,
            "credits_wht": credits_wht, "credits_foreign": credits_foreign,
            "rebates": rebates, "provisional_tax_paid": provisional_tax_paid,
            "net_tax_payable": net_tax_after_provisional
        }

        st.success("Computation complete — see summary below.")
        st.metric("Taxable Income (after capital allowances & exemptions)", f"UGX {adjusted_taxable_income:,.2f}")
        st.metric("Gross Tax (before credits)", f"UGX {gross_tax:,.2f}")
        st.metric("Net Tax Payable (after credits & rebates)", f"UGX {net_tax_payable:,.2f}")
        st.metric("Net Tax Payable (after provisional payments)", f"UGX {net_tax_after_provisional:,.2f}")

        if st.button("💾 Save Computation to History (DB)", key="t3_btn_save_history"):
            row = {
                "client_name": client_name,
                "taxpayer_type": taxpayer_type,
                "year": int(tax_year),
                "period": period_label,
                "revenue": revenue, "cogs": cogs, "opex": opex,
                "other_income": other_income, "other_expenses": other_expenses,
                "pbit": pbit,
                "capital_allowances": capital_allowances, "exemptions": exemptions,
                "taxable_income": adjusted_taxable_income, "gross_tax": gross_tax,
                "credits_wht": credits_wht, "credits_foreign": credits_foreign,
                "rebates": rebates, "net_tax_payable": net_tax_after_provisional,
                "metadata_json": json.dumps({"TIN": tin}),
                "created_at": datetime.utcnow().isoformat()
            }
            save_history(row)
            load_history_cached.clear()  # refresh cache
            st.success("Saved to history.")

# ----------------------------
# Tab 4: Dashboard
# ----------------------------
with tab4:
    st.subheader("📊 Multi-Year History Dashboard")
    
    # Load historical tax data
    data_file = Path("data/tax_history.csv")
    if data_file.exists():
        df_history = pd.read_csv(data_file)
    else:
        st.warning("Historical data file not found. Please upload 'tax_history.csv' in the 'data/' folder.")
        df_history = pd.DataFrame(columns=["Client", "Year", "Tax Liability", "Sector", "Revenue"])

    # Filter by client
    client_filter = st.text_input("Filter by client name:")
    if client_filter:
        df_filtered = df_history[df_history["Client"].str.contains(client_filter, case=False, na=False)]
    else:
        df_filtered = df_history.copy()

    # Filter by year range
    min_year = int(df_history["Year"].min()) if not df_history.empty else 2000
    max_year = int(df_history["Year"].max()) if not df_history.empty else 2025
    year_range = st.slider("Select year range:", min_value=min_year, max_value=max_year, value=(min_year, max_year))
    df_filtered = df_filtered[(df_filtered["Year"] >= year_range[0]) & (df_filtered["Year"] <= year_range[1])]

    # Display filtered data
    st.dataframe(df_filtered.reset_index(drop=True))

    # Plot total tax liability over time
    if not df_filtered.empty:
        tax_trend = df_filtered.groupby("Year")["Tax Liability"].sum().reset_index()
        st.line_chart(tax_trend.rename(columns={"Tax Liability": "Total Tax Liability"}).set_index("Year"))

        # Sector-wise breakdown
        sector_breakdown = df_filtered.groupby("Sector")["Tax Liability"].sum().sort_values(ascending=False)
        st.bar_chart(sector_breakdown)
    else:
        st.info("No data available for the selected filters.")
    
    # Download filtered data
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download filtered data as CSV",
        data=csv,
        file_name="filtered_tax_history.csv",
        mime="text/csv"
    )

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
if st.session_state.get("authentication_status"):
    if st.sidebar.button("Logout"):
        st.session_state["authentication_status"] = False
        st.experimental_rerun()

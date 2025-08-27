import os
import io
import re
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
import requests

# Login + Subscription + Tax History Module

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="TaxIntellilytics — Income Tax (Uganda)", layout="wide")

# ================================
# USER DATABASE (login + subscription)
# ================================
def init_user_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT, subscription_expiry TEXT)''')
    conn.commit()
    conn.close()

def add_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO users VALUES (?,?,?)",
              (username, password, None))
    conn.commit()
    conn.close()

def update_subscription(username, days=30):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    expiry = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
    c.execute("UPDATE users SET subscription_expiry=? WHERE username=?", (expiry, username))
    conn.commit()
    conn.close()

def check_subscription(username):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT subscription_expiry FROM users WHERE username=?", (username,))
    result = c.fetchone()
    conn.close()
    if result and result[0]:
        expiry_date = datetime.strptime(result[0], "%Y-%m-%d")
        return expiry_date >= datetime.now()
    return False

# ================================
# FLUTTERWAVE PAYMENT
# ================================
FLUTTERWAVE_SECRET_KEY = "FLWSECK_TEST-xxxxxxxxxxxx"  # replace with your sandbox key
FLUTTERWAVE_PUBLIC_KEY = "FLWPUBK_TEST-xxxxxxxxxxxx"

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
        "redirect_url": "http://localhost:8501",  # change to deployed app URL
        "customer": {"email": f"{username}@example.com", "name": username},
        "customizations": {"title": "TaxIntellilytics Subscription", "description": "Access premium tax analytics"}
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["data"]["link"]
    else:
        st.error("Payment initialization failed.")
        return None

# ================================
# TAX HISTORY DATABASE
# ================================
DB_PATH = "taxintellilytics_history.sqlite"

def init_tax_db():
    with sqlite3.connect(DB_PATH) as conn:
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
    with sqlite3.connect(DB_PATH) as conn:
        cols = ",".join(row.keys())
        placeholders = ",".join(["?"] * len(row))
        conn.execute(f"INSERT INTO income_tax_history ({cols}) VALUES ({placeholders})", list(row.values()))
        conn.commit()

@st.cache_data(ttl=600)
def load_history_cached(client_filter: str = "") -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM income_tax_history ORDER BY year DESC, created_at DESC", conn)
    if client_filter and not df.empty:
        df = df[df["client_name"].str.contains(client_filter, case=False, na=False)]
    return df

# ================================
# TAXINTELLILYTICS DASHBOARD
# ================================
def show_tax_module():
    st.title("💼 TaxIntellilytics — Income Tax (Uganda)")
    st.caption("Automating, Analyzing, and Advancing Tax Compliance in Uganda — with Audit Controls")

    st.subheader("📊 Income Tax Dashboard")
    st.write("Here goes your tax computation, audit controls, and history features.")

    # Example: Load history
    df = load_history_cached()
    if not df.empty:
        st.dataframe(df)

# ================================
# MAIN APP FLOW
# ================================
def main():
    # Init databases
    init_user_db()
    init_tax_db()

    # Demo credentials (replace with DB-driven later)
    usernames = ["user1", "user2"]
    passwords = ["12345", "password"]
    for u, p in zip(usernames, passwords):
        add_user(u, stauth.Hasher([p]).generate()[0])

    # Authentication
    authenticator = stauth.Authenticate(
        {"usernames": {u: {"name": u, "password": stauth.Hasher([p]).generate()[0]} for u, p in zip(usernames, passwords)}},
        "my_cookie", "my_signature_key", cookie_expiry_days=1
    )

    name, authentication_status, username = authenticator.login("Login", "main")

    if authentication_status == False:
        st.error("Username/password is incorrect")
    elif authentication_status == None:
        st.warning("Please enter your username and password")
    else:
        authenticator.logout("Logout", "sidebar")
        st.sidebar.write(f"Welcome {name} 👋")

        # Subscription check
        if check_subscription(username):
            st.success("✅ Subscription active! Access granted.")
            show_tax_module()
        else:
            st.warning("🚨 You need an active subscription to access TaxIntellilytics.")
            if st.button("Subscribe with Flutterwave"):
                link = create_payment_link(username)
                if link:
                    st.markdown(f"[Click here to pay via Flutterwave]({link})")

if __name__ == "__main__":
    main()

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="TaxIntellilytics — Income Tax (Uganda)", layout="wide")
st.title("💼 TaxIntellilytics — Income Tax (Uganda)")
st.caption("Automating, Analyzing, and Advancing Tax Compliance in Uganda — with Audit Controls")

# ----------------------------
# DB Initialization (SQLite)
# ----------------------------
DB_PATH = "taxintellilytics_history.sqlite"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
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
init_db()

def save_history(row: dict):
    with sqlite3.connect(DB_PATH) as conn:
        cols = ",".join(row.keys())
        placeholders = ",".join(["?"] * len(row))
        conn.execute(f"INSERT INTO income_tax_history ({cols}) VALUES ({placeholders})", list(row.values()))
        conn.commit()

@st.cache_data(ttl=600)
def load_history_cached(client_filter: str = "") -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM income_tax_history ORDER BY year DESC, created_at DESC", conn)
    if client_filter and not df.empty:
        df = df[df["client_name"].str.contains(client_filter, case=False, na=False)]
    return df# ----------------------------
# QuickBooks helpers (cached)
# ----------------------------
@st.cache_resource
def qb_is_available_cached():
    try:
        import intuitlib, quickbooks  # noqa
        return True
    except Exception:
        return False

def qb_env_ready():
    required = ["QB_CLIENT_ID", "QB_CLIENT_SECRET", "QB_REDIRECT_URI", "QB_ENVIRONMENT", "QB_REALM_ID"]
    return all(os.getenv(k) for k in required)

def qb_connect_button():
    st.subheader("🔗 QuickBooks Connection (Optional)")
    if not qb_is_available_cached():
        st.info("QuickBooks SDK not installed. Install `intuit-oauth` and `python-quickbooks` to enable.")
        return None
    if not qb_env_ready():
        st.warning("Set env vars QB_CLIENT_ID, QB_CLIENT_SECRET, QB_REDIRECT_URI, QB_ENVIRONMENT, QB_REALM_ID to enable OAuth2.")
        return None
    st.write("Environment ready ✅. (This is a simulated button for demo.)")
    if st.button("Fetch P&L from QuickBooks (Simulated)", key="t1_btn_qb_fetch"):
        data = {
            "Account": ["Income:Sales", "Income:Other Income", "COGS", "Expenses:Rent", "Expenses:Salaries"],
            "Amount": [250_000_000, 10_000_000, 90_000_000, 30_000_000, 60_000_000],
        }
        df = pd.DataFrame(data)
        st.success("Fetched P&L (simulated).")
        st.dataframe(df.head(50), use_container_width=True)
        return df
    return None

# ----------------------------
# File Parsing & Auto-map (cached)
# ----------------------------
@st.cache_data
def parse_financial_file_bytes(uploaded_bytes: bytes, filename: str) -> pd.DataFrame:
    buf = io.BytesIO(uploaded_bytes)
    if filename.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(buf)
    else:
        buf.seek(0)
        return pd.read_csv(buf)

@st.cache_data
def auto_map_pl_cached(df_json: str) -> Tuple[float, float, float, float, float]:
    # We pass a JSON serialized version of df to caching to avoid caching raw DataFrame object
    df = pd.read_json(df_json)
    cols = {c.lower().strip(): c for c in df.columns}
    revenue = df[cols.get("revenue")].sum() if "revenue" in cols else 0.0
    cogs = df[cols.get("cogs")].sum() if "cogs" in cols else 0.0
    opex = df[cols.get("operating_expenses")].sum() if "operating_expenses" in cols else 0.0
    other_income = df[cols.get("other_income")].sum() if "other_income" in cols else 0.0
    other_expenses = df[cols.get("other_expenses")].sum() if "other_expenses" in cols else 0.0

    if "account" in cols and "amount" in cols:
        tmp = df[[cols["account"], cols["amount"]]].copy()
        tmp.columns = ["Account", "Amount"]
        revenue += tmp[tmp["Account"].str.contains("income|sales|revenue", case=False, na=False)]["Amount"].sum()
        cogs += tmp[tmp["Account"].str.contains("cogs|cost of goods", case=False, na=False)]["Amount"].sum()
        opex += tmp[tmp["Account"].str.contains("expense|utilities|rent|salary|transport|admin", case=False, na=False)]["Amount"].sum()
        other_income += tmp[tmp["Account"].str.contains("other income|gain", case=False, na=False)]["Amount"].sum()
        other_expenses += tmp[tmp["Account"].str.contains("other expense|loss", case=False, na=False)]["Amount"].sum()

    return float(revenue), float(cogs), float(opex), float(other_income), float(other_expenses)

# ----------------------------
# Tax computation functions
# ----------------------------
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
    
# ----------------------------
# URA Schemas & Validator
# ----------------------------

URA_SCHEMAS = {
    "DT-2001": {  # Income Tax Return - Individual with Business Income
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
    "DT-2002": {  # Income Tax Return - Non-Individual (Company)
        "title": "Income Tax Return Form for Non-Individual",
        "fields": [
            ("TIN", "str"),
            ("Entity Name", "str"),
            ("Period", "str"),
            ("Year", "int"),
            ("Gross Turnover (UGX)", "float"),
            ("COGS (UGX)", "float"),
            ("Operating Expenses (UGX)", "float"),
            ("Other Income (UGX)", "float"),
            ("Other Expenses (UGX)", "float"),
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
    "DT-2003": {  # Provisional Income Tax Return - Individual
        "title": "Income Tax Provisional Return Form for Individual",
        "fields": [
            ("TIN", "str"),
            ("Taxpayer Name", "str"),
            ("Period", "str"),
            ("Year", "int"),
            ("Estimated Business Income (UGX)", "float"),
            ("Estimated Deductions (UGX)", "float"),
            ("Estimated Capital Allowances (UGX)", "float"),
            ("Estimated Exemptions (UGX)", "float"),
            ("Estimated Taxable Income (UGX)", "float"),
            ("Provisional Tax (UGX)", "float"),
            ("WHT Credits (UGX)", "float"),
            ("Net Provisional Tax Payable (UGX)", "float"),
        ],
    },
    "DT-2004": {  # Income Tax Return - Partnership
        "title": "Income Tax Return Form for Partnership",
        "fields": [
            ("TIN", "str"),
            ("Partnership Name", "str"),
            ("Period", "str"),
            ("Year", "int"),
            ("Gross Income (UGX)", "float"),
            ("Allowable Expenses (UGX)", "float"),
            ("Capital Allowances (UGX)", "float"),
            ("Exemptions (UGX)", "float"),
            ("Taxable Partnership Income (UGX)", "float"),
            ("Gross Tax (UGX)", "float"),
            ("WHT Credits (UGX)", "float"),
            ("Net Tax Payable (UGX)", "float"),
        ],
    },
}


def validate_and_build_return(form_code: str, payload: dict) -> pd.DataFrame:
    """
    Validate a payload against the URA schema and return a pandas DataFrame.

    Args:
        form_code (str): URA form code, must exist in URA_SCHEMAS
        payload (dict): Dictionary containing field values

    Returns:
        pd.DataFrame: Single-row DataFrame ready for Excel/CSV export

    Raises:
        ValueError: If the form code is unsupported, a field is missing, or has invalid type
    """
    if form_code not in URA_SCHEMAS:
        raise ValueError(f"Unsupported form code: {form_code}")

    schema = URA_SCHEMAS[form_code]["fields"]
    validated = {}

    for field, ftype in schema:
        if field not in payload:
            raise ValueError(f"Missing required field: {field}")

        val = payload[field]

        try:
            if ftype == "int":
                validated[field] = int(val)
            elif ftype == "float":
                validated[field] = float(val)
            else:
                # Clean up string fields by stripping extra whitespace
                validated[field] = str(val).strip() if val is not None else ""
        except Exception:
            raise ValueError(f"Invalid type for field '{field}', expected {ftype}, got value: {val}")

    return pd.DataFrame([validated])

# ----------------------------
# AUDIT & CONTROL ACCOUNTS
# ----------------------------
DEFAULT_CONTROL_MAP = pd.DataFrame([
    # Category, NormalBalance ("Debit" or "Credit"), Regex patterns (comma-separated), Materiality (UGX)
    ["Cash & Bank", "Debit", r"(?i)\bcash\b|bank|current account|cash at hand", 50_000],
    ["Accounts Receivable", "Debit", r"(?i)\bar\b|trade receivable|debtors", 50_000],
    ["Inventory", "Debit", r"(?i)\binventory\b|stock", 50_000],
    ["Accounts Payable", "Credit", r"(?i)\bap\b|trade payable|creditors", 50_000],
    ["VAT Payable", "Credit", r"(?i)\bvat\b|output vat|vat payable", 50_000],
    ["VAT Receivable", "Debit", r"(?i)input vat|vat receivable", 50_000],
    ["PAYE Payable", "Credit", r"(?i)paye\b|pay as you earn", 50_000],
    ["WHT Receivable", "Debit", r"(?i)withholding tax (receivable|asset)|wht receivable|wht asset", 50_000],
    ["Income Tax Payable", "Credit", r"(?i)income tax payable|corporation tax payable", 50_000],
    ["Share Capital", "Credit", r"(?i)share capital|stated capital", 50_000],
    ["Retained Earnings", "Credit", r"(?i)retained earnings|accumulated (profit|loss)", 50_000],
], columns=["Category", "NormalBalance", "Patterns", "MaterialityUGX"])

def harmonize_tb(df: pd.DataFrame) -> pd.DataFrame:
    """Accept TB with Debit/Credit columns or a single Amount. Return ['Account','Debit','Credit','Amount']."""
    cols = {c.lower().strip(): c for c in df.columns}
    df2 = df.copy()
    # rename common fields
    for wanted in ["account", "description"]:
        if wanted in cols and "account" not in df2.columns:
            df2.rename(columns={cols[wanted]: "Account"}, inplace=True)

    # Cases:
    # 1) Has both debit & credit columns
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
        # 2) single Amount column (signed)
        amt_col = None
        for c in df2.columns:
            if c.lower() in ["amount", "balance", "closing balance", "ending balance", "net"]:
                amt_col = c
                break
        if amt_col is None:
            # try last numeric column
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

    # keep minimal
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
        # Expected sign based on normal balance
        expected_sign = 1 if nb.lower().startswith("debit") else -1
        signed_ok = (amt == 0) or (np.sign(amt) == expected_sign)
        exception = 0.0
        if not signed_ok:
            exception = abs(amt)  # sign mismatch is fully exceptional
        elif abs(amt) < materiality:
            exception = 0.0
        else:
            exception = 0.0  # within sign; no numeric threshold breach (threshold only used for presence)
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
    # Basic heuristic: income/expense by keywords
    acc = tb_df["Account"].astype(str)
    is_income = acc.str.contains(r"(?i)\b(income|revenue|sales|gain)\b")
    is_cogs = acc.str.contains(r"(?i)cogs|cost of goods|cost of sales")
    is_opex = acc.str.contains(r"(?i)expense|utilities|rent|salary|transport|admin|repairs|maintenance")
    is_other_income = acc.str.contains(r"(?i)other income|finance income|interest income|dividend income|gain")
    is_other_exp = acc.str.contains(r"(?i)other expense|finance cost|interest expense|loss")

    # Amount sign convention: Debit positive, Credit negative (from harmonize_tb)
    amt = tb_df["Amount"].astype(float)

    revenue = -re_sum(amt[is_income])  # credits -> positive revenue
    cogs = re_sum(amt[is_cogs])        # debits -> positive cost
    opex = re_sum(amt[is_opex])
    other_income = -re_sum(amt[is_other_income])
    other_expenses = re_sum(amt[is_other_exp])

    return dict(revenue=revenue, cogs=cogs, opex=opex, other_income=other_income, other_expenses=other_expenses)

def audit_findings(tb_df: pd.DataFrame,
                   control_map: pd.DataFrame,
                   mapped_pl: Dict[str, float],
                   materiality_total_ugx: float = 100_000.0) -> Dict[str, pd.DataFrame]:
    # 1) Trial balance integrity
    debit_total = re_sum(tb_df["Debit"])
    credit_total = re_sum(tb_df["Credit"])
    tb_diff = round(debit_total - credit_total, 2)

    tb_integrity = pd.DataFrame([{
        "TotalDebit": debit_total,
        "TotalCredit": credit_total,
        "Difference(should be 0)": tb_diff,
        "Pass": abs(tb_diff) < 1e-2
    }])

    # 2) Control accounts match & sign tests
    ctrl = match_control_amounts(tb_df, control_map)

    # 3) P&L reconciliation (TB-derived vs user-mapped)
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

    # 4) Exceptions summary
    exceptions = []
    if not tb_integrity["Pass"].iloc[0]:
        exceptions.append({"Area": "TB Integrity", "Issue": "Debits != Credits", "Amount": tb_diff})

    # control account sign issues
    for _, r in ctrl.iterrows():
        if not bool(r["SignOK"]):
            exceptions.append({
                "Area": f"Control: {r['Category']}",
                "Issue": f"Sign mismatch (Expected {r['ExpectedSign']})",
                "Amount": float(r["MatchedAmount"])
            })
        elif abs(float(r["MatchedAmount"])) >= float(r["MaterialityUGX"]):
            # it's OK if large but sign ok — not an exception per se; we log notable balances separately
            pass

    # P&L reconciliation material deltas
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
    qb_df = qb_connect_button()
    uploaded = st.file_uploader(
        "Upload P&L / Trial Balance (CSV or Excel)",
        type=["csv", "xlsx"],
        help="Upload a Profit & Loss or Trial Balance export from your accounting system",
        key="t1_file_uploader"
    )

    df = None
    if qb_df is not None:
        df = qb_df
    if uploaded is not None:
        uploaded_bytes = uploaded.getvalue()
        try:
            df = parse_financial_file_bytes(uploaded_bytes, uploaded.name)
        except Exception as e:
            st.error(f"Failed to parse file: {e}")
            df = None

    if df is not None and not df.empty:
        st.session_state["pl_df"] = df
        st.write("### Preview (first 50 rows)")
        st.dataframe(df.head(50), use_container_width=True)
        if st.button("Auto-Map P&L (fast)", key="t1_btn_automap"):
            df_json = df.head(10000).to_json()
            revenue, cogs, opex, other_income, other_expenses = auto_map_pl_cached(df_json)
            st.session_state["mapped_values"] = {
                "revenue": revenue, "cogs": cogs, "opex": opex,
                "other_income": other_income, "other_expenses": other_expenses
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
    client_filter = st.text_input("Filter by client name (optional)", "", key="t4_client_filter")
    hist = load_history_cached(client_filter)
    if hist.empty:
        st.info("No saved history yet.")
    else:
        st.write("Showing latest 200 records (use filter to narrow).")
        st.dataframe(hist.head(200), use_container_width=True)
        st.markdown("#### Net Tax by Year")
        pivot = hist.groupby(["year"])["net_tax_payable"].sum().reset_index()
        if not pivot.empty:
            st.line_chart(pivot.rename(columns={"net_tax_payable": "Net Tax Payable"}).set_index("year"))
        st.markdown("#### Taxable Income vs Gross Tax (latest 30)")
        if len(hist) > 0:
            chart_df = hist.head(30).set_index("created_at")[["taxable_income", "gross_tax"]]
            st.bar_chart(chart_df)

# ----------------------------
# Tab 5: Export & URA Forms
# ----------------------------
with tab5:
    st.subheader("📤 URA Return CSV / Excel (DT-2001 / DT-2002) with Validation")
    last = st.session_state.get("last_computation", {})
    suggested_client = last.get("client_name", "")
    suggested_year = int(last.get("year", tax_year))
    suggested_period = last.get("period", period_label)
    suggested_taxable = float(last.get("taxable_income", 0.0))
    suggested_gross = float(last.get("gross_tax", 0.0))

    st.markdown("Fill the required fields to build a URA-compliant CSV/Excel.")
    form_code = "DT-2002" if taxpayer_type.lower() == "company" else "DT-2001"
    st.info(f"Selected Form: **{form_code}**")

    TIN_input = st.text_input("TIN (required)", value=last.get("TIN", ""), key="t5_tin_input")

    if form_code == "DT-2001":
        taxpayer_name = st.text_input("Taxpayer Name", value=suggested_client, key="t5_taxpayer_name")
        business_income = st.number_input("Business Income (UGX)", min_value=0.0, value=suggested_taxable, format="%.2f", key="t5_biz_income")
        allowable_deductions = st.number_input("Allowable Deductions (UGX)", min_value=0.0, value=float(last.get("total_allowables", 0.0)), format="%.2f", key="t5_allowable_deductions")
        capital_allowances_f = st.number_input("Capital Allowances (UGX)", min_value=0.0, value=float(last.get("capital_allowances", 0.0)), format="%.2f", key="t5_cap_allowances")
        exemptions_f = st.number_input("Exemptions (UGX)", min_value=0.0, value=float(last.get("exemptions", 0.0)), format="%.2f", key="t5_exemptions")
        gross_tax_f = st.number_input("Gross Tax (UGX)", min_value=0.0, value=suggested_gross, format="%.2f", key="t5_gross_tax")
        wht_f = st.number_input("WHT Credits (UGX)", min_value=0.0, value=float(last.get("credits_wht", 0.0)), format="%.2f", key="t5_wht")
        foreign_f = st.number_input("Foreign Tax Credit (UGX)", min_value=0.0, value=float(last.get("credits_foreign", 0.0)), format="%.2f", key="t5_ftc")
        rebates_f = st.number_input("Rebates (UGX)", min_value=0.0, value=float(last.get("rebates", 0.0)), format="%.2f", key="t5_rebates")

        payload = {
            "TIN": TIN_input,
            "Taxpayer Name": taxpayer_name,
            "Period": suggested_period,
            "Year": suggested_year,
            "Business Income (UGX)": business_income,
            "Allowable Deductions (UGX)": allowable_deductions,
            "Capital Allowances (UGX)": capital_allowances_f,
            "Exemptions (UGX)": exemptions_f,
            "Taxable Income (UGX)": max(0.0, business_income - allowable_deductions - capital_allowances_f - exemptions_f),
            "Gross Tax (UGX)": gross_tax_f,
            "WHT Credits (UGX)": wht_f,
            "Foreign Tax Credit (UGX)": foreign_f,
            "Rebates (UGX)": rebates_f,
            "Net Tax Payable (UGX)": max(0.0, gross_tax_f - wht_f - foreign_f - rebates_f)
        }

    else:  # DT-2002
        entity_name = st.text_input("Entity Name", value=suggested_client, key="t5_entity_name")
        gross_turnover = st.number_input("Gross Turnover (UGX)", min_value=0.0, value=float(last.get("revenue", 0.0)), format="%.2f", key="t5_gturnover")
        cogs_f = st.number_input("COGS (UGX)", min_value=0.0, value=float(last.get("cogs", 0.0)), format="%.2f", key="t5_cogs")
        opex_f = st.number_input("Operating Expenses (UGX)", min_value=0.0, value=float(last.get("opex", 0.0)), format="%.2f", key="t5_opex")
        other_income_f = st.number_input("Other Income (UGX)", min_value=0.0, value=float(last.get("other_income", 0.0)), format="%.2f", key="t5_oincome")
        other_expenses_f = st.number_input("Other Expenses (UGX)", min_value=0.0, value=float(last.get("other_expenses", 0.0)), format="%.2f", key="t5_oexpense")
        capital_allowances_f = st.number_input("Capital Allowances (UGX)", min_value=0.0, value=float(last.get("capital_allowances", 0.0)), format="%.2f", key="t5_cap_allowances_c")
        exemptions_f = st.number_input("Exemptions (UGX)", min_value=0.0, value=float(last.get("exemptions", 0.0)), format="%.2f", key="t5_exemptions_c")
        gross_tax_f = st.number_input("Gross Tax (UGX)", min_value=0.0, value=suggested_gross, format="%.2f", key="t5_gross_tax_c")
        wht_f = st.number_input("WHT Credits (UGX)", min_value=0.0, value=float(last.get("credits_wht", 0.0)), format="%.2f", key="t5_wht_c")
        foreign_f = st.number_input("Foreign Tax Credit (UGX)", min_value=0.0, value=float(last.get("credits_foreign", 0.0)), format="%.2f", key="t5_ftc_c")
        rebates_f = st.number_input("Rebates (UGX)", min_value=0.0, value=float(last.get("rebates", 0.0)), format="%.2f", key="t5_rebates_c")

        taxable_income_calc = max(0.0, (gross_turnover + other_income_f) - (cogs_f + opex_f + other_expenses_f) - capital_allowances_f - exemptions_f)
        payload = {
            "TIN": TIN_input,
            "Entity Name": entity_name,
            "Period": suggested_period,
            "Year": suggested_year,
            "Gross Turnover (UGX)": gross_turnover,
            "COGS (UGX)": cogs_f,
            "Operating Expenses (UGX)": opex_f,
            "Other Income (UGX)": other_income_f,
            "Other Expenses (UGX)": other_expenses_f,
            "Capital Allowances (UGX)": capital_allowances_f,
            "Exemptions (UGX)": exemptions_f,
            "Taxable Income (UGX)": taxable_income_calc,
            "Gross Tax (UGX)": gross_tax_f,
            "WHT Credits (UGX)": wht_f,
            "Foreign Tax Credit (UGX)": foreign_f,
            "Rebates (UGX)": rebates_f,
            "Net Tax Payable (UGX)": max(0.0, gross_tax_f - wht_f - foreign_f - rebates_f)
        }

    if st.button("✅ Validate & Build CSV / Excel", key="t5_btn_build"):
        try:
            df_return = validate_and_build_return(form_code, payload)
            st.success("Validation passed. Download your URA return below.")
            csv_bytes = df_return.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download URA Return CSV",
                data=csv_bytes,
                file_name=f"{form_code}_{payload.get('Year')}_{payload.get('TIN','')}.csv",
                mime="text/csv",
                key="t5_dl_csv"
            )
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_return.to_excel(writer, index=False, sheet_name=form_code)
            st.download_button(
                label="📥 Download URA Return Excel",
                data=buffer.getvalue(),
                file_name=f"{form_code}_{payload.get('Year')}_{payload.get('TIN','')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="t5_dl_xlsx"
            )
            st.dataframe(df_return, use_container_width=True)
        except Exception as e:
            st.error(f"Validation failed: {e}")
# ----------------------------
# Tab 6: Audit & Controls
# ----------------------------
with tab6:
    st.subheader("🔎 Audit & Control Accounts")
    st.markdown("""
    Upload a **Trial Balance** (TB). This tool will:
    1) Check TB integrity (Debits = Credits),
    2) Match & test sign expectations for control accounts (editable map in sidebar),
    3) Reconcile TB-derived P&L vs your mapped P&L values (materiality in sidebar).
    """)

    # Upload Trial Balance
    tb_file = st.file_uploader(
        "Upload Trial Balance (CSV/Excel)", 
        type=["csv", "xlsx"], 
        key="tb_upload"
    )

    # Materiality threshold input
    materiality = st.sidebar.number_input(
        "Materiality Threshold (UGX)", 
        min_value=0.0, 
        value=1000000.0, 
        step=100000.0,
        key="materiality_threshold"
    )

    # Example control accounts map
    st.sidebar.markdown("### 🔧 Control Accounts Map")
    control_accounts = {
        "Cash": "Debit",
        "Bank": "Debit",
        "Accounts Receivable": "Debit",
        "Accounts Payable": "Credit",
        "Revenue": "Credit",
        "Expenses": "Debit"
    }

    # Editable control account mapping in sidebar
    user_control_map = {}
    for i, (acc, expected_sign) in enumerate(control_accounts.items()):
        user_choice = st.sidebar.selectbox(
            f"{acc} Expected Sign",
            ["Debit", "Credit"],
            index=0 if expected_sign == "Debit" else 1,
            key=f"ctrl_{acc}_{i}"
        )
        user_control_map[acc] = user_choice

    if tb_file:
        import pandas as pd
        import numpy as np

        # Load the trial balance
        try:
            if tb_file.name.endswith(".csv"):
                tb_df = pd.read_csv(tb_file)
            else:
                tb_df = pd.read_excel(tb_file)

            st.write("✅ Trial Balance Preview:")
            st.dataframe(tb_df.head())

            # Check TB integrity
            total_debits = tb_df["Debit"].sum() if "Debit" in tb_df.columns else 0
            total_credits = tb_df["Credit"].sum() if "Credit" in tb_df.columns else 0

            if np.isclose(total_debits, total_credits):
                st.success(f"Trial Balance is balanced. Total Debits = {total_debits}, Total Credits = {total_credits}")
            else:
                st.error(f"Trial Balance is NOT balanced! Debits = {total_debits}, Credits = {total_credits}")

            # Control accounts check
            st.markdown("### Control Accounts Sign Check")
            for acc, expected_sign in user_control_map.items():
                if acc in tb_df["Account"].values:
                    acc_balance = tb_df.loc[tb_df["Account"] == acc, "Debit"].sum() - tb_df.loc[tb_df["Account"] == acc, "Credit"].sum()
                    sign = "Debit" if acc_balance >= 0 else "Credit"
                    if sign == expected_sign:
                        st.success(f"{acc}: OK ({sign})")
                    else:
                        st.warning(f"{acc}: Mismatch! Expected {expected_sign}, Found {sign}")
                else:
                    st.info(f"{acc}: Account not in TB")

            # P&L reconciliation (optional)
            st.markdown("### P&L Reconciliation")
            tb_pl_total = tb_df.loc[tb_df["Account"].isin(["Revenue", "Expenses"]), "Debit"].sum() - tb_df.loc[tb_df["Account"].isin(["Revenue", "Expenses"]), "Credit"].sum()
            if abs(tb_pl_total) <= materiality:
                st.success(f"P&L within materiality ({tb_pl_total} UGX)")
            else:
                st.warning(f"P&L outside materiality ({tb_pl_total} UGX)")

        except Exception as e:
            st.error(f"Error processing TB: {e}")
            
import streamlit as st
import pandas as pd

# ----------------------------
# End of App
# ----------------------------
st.markdown("---")
st.markdown("© 2025 TaxIntellilytics | Designed for automated tax computation, simulation, and audit compliance in Uganda.")

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
# Paths & DB Initialization
# ---------------------------
DB_DIR = "/tmp/taxintellilytics"
USERS_DB = f"{DB_DIR}/users.db"
HISTORY_DB = f"{DB_DIR}/taxintellilytics_history.sqlite"
os.makedirs(DB_DIR, exist_ok=True)

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

init_user_db()
init_history_db()

# ---------------------------
# Authentication Helpers
# ---------------------------
def rand_salt():
    return hashlib.sha256(os.urandom(16)).hexdigest()[:16]

def hash_with_salt(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

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

SUBSCRIPTION_PLANS = {
    "Basic - 500,000 UGX/month": {"amount": 500_000, "days": 30},
    "Standard - 1,000,000 UGX/month": {"amount": 1_000_000, "days": 30},
    "Premium - 1,500,000 UGX/month": {"amount": 1_500_000, "days": 30},
    "Annual - 10% of monthly × 12": {"amount": 500_000*12*0.90, "days": 365}
}

# ---------------------------
# Streamlit Page Setup
# ---------------------------
st.set_page_config(page_title="TaxIntellilytics", layout="wide")
st.title("💼 TaxIntellilytics — Uganda Tax Compliance Toolkit")

# ---------------------------
# Session State Defaults
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
# Authentication UI
# ---------------------------
def show_auth_ui():
    tab_login, tab_signup = st.tabs(["Login / Renew", "Sign Up"])
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
                    st.session_state["subscription_active"] = check_subscription(login_username)
                    st.session_state["plan"] = plan
                    st.success(f"Welcome {login_username} 👋")
                    if not st.session_state["subscription_active"]:
                        st.warning("🚨 Your subscription is inactive. Choose a plan to activate below.")
                else:
                    st.error("Incorrect password")
        if st.session_state.get("authenticated") and not st.session_state["subscription_active"]:
            selected_plan = st.selectbox(
                "Select a subscription plan", 
                list(SUBSCRIPTION_PLANS.keys()), 
                key="login_plan_tab"
            )
            if st.button("Subscribe via MTN/Airtel", key="pay_btn_tab"):
                plan_info = SUBSCRIPTION_PLANS[selected_plan]
                pay_link = f"https://pay.example.com/{st.session_state['current_user']}?amount={plan_info['amount']}"
                st.markdown(f"[Click here to pay via MTN/Airtel]({pay_link})")
                st.session_state["payment_pending"] = True
                st.session_state["pending_plan"] = selected_plan
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
                pay_link = f"https://pay.example.com/{signup_username}?amount={plan_info['amount']}"
                st.markdown(f"[Click here to pay via MTN/Airtel]({pay_link})")
                st.session_state["payment_pending"] = True
                st.session_state["pending_plan"] = signup_plan

if not st.session_state["authenticated"]:
    show_auth_ui()
    st.stop()

if st.session_state.get("payment_pending"):
    st.info("Waiting for payment confirmation...")
    if st.button("Confirm Payment (Demo)"):
        plan_info = SUBSCRIPTION_PLANS[st.session_state["pending_plan"]]
        update_subscription(st.session_state["current_user"], days=plan_info["days"], plan=st.session_state["pending_plan"])
        st.session_state["subscription_active"] = True
        st.session_state["payment_pending"] = False
        st.success(f"✅ Subscription activated for {st.session_state['pending_plan']}!")

if st.session_state["authenticated"] and st.session_state["subscription_active"]:
    st.sidebar.write(f"Welcome {st.session_state['current_user']} 👋")
    if st.sidebar.button("Logout"):
        for k in ["authenticated", "current_user", "subscription_active", "plan", "payment_pending", "pending_plan"]:
            st.session_state[k] = False if k == "authenticated" else None
        st.experimental_rerun()

# ---------------------------
# Sidebar Configuration
# ---------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    taxpayer_type = st.selectbox("Taxpayer Type", ["company", "individual"], key="sb_taxpayer_type")
    tax_year = st.number_input("Year", min_value=2000, max_value=datetime.now().year, value=datetime.now().year, step=1, key="sb_tax_year")
    period_label = st.text_input("Period label (e.g., FY2024/25)", value=f"FY{tax_year}", key="sb_period_label")
    company_rate = st.number_input("Company Income Tax Rate", min_value=0.0, max_value=1.0, value=0.30, step=0.01, key="sb_company_rate")
    audit_materiality = st.number_input("P&L Reconciliation Materiality", min_value=0.0, value=100_000.0, step=10_000.0, key="sb_audit_mat")

# ---------------------------
# Main Tabs
# ---------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "1) Data Import", "2) P&L Mapping", "3) Compute & Credits", "4) Dashboard", "5) Export & URA Forms",
    "6) Audit & Controls", "7) VAT", "8) Presumptive Tax", "9) Payroll", "10) Withholding Tax (WHT)", "11) Transfer Pricing"
])

# ---------------------------
# Tab 1: Data Import
# ---------------------------
with tab1:
    st.subheader("📂 Upload Financials (CSV/XLSX) or Connect to QuickBooks (Optional)")

    # Simulated QuickBooks connection button
    def qb_connect_button():
        import pandas as pd
        return pd.DataFrame()  # empty DataFrame as placeholder

    qb_df = qb_connect_button()

    uploaded = st.file_uploader(
        "Upload P&L / Trial Balance (CSV or Excel)",
        type=["csv", "xlsx"],
        help="Upload a Profit & Loss or Trial Balance export from your accounting system",
        key="tab1_file_uploader"
    )

    df = None
    if qb_df is not None and not qb_df.empty:
        df = qb_df
    if uploaded is not None:
        uploaded_bytes = uploaded.getvalue()
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(uploaded_bytes))
            elif uploaded.name.endswith(".xlsx"):
                df = pd.read_excel(io.BytesIO(uploaded_bytes))
            else:
                raise ValueError("Unsupported file type")
        except Exception as e:
            st.error(f"Failed to parse file: {e}")
            df = None

    if df is not None and not df.empty:
        st.session_state["pl_df"] = df
        st.write("### Preview (first 50 rows)")
        st.dataframe(df.head(50), use_container_width=True)
        if st.button("Auto-Map P&L (fast)", key="tab1_btn_automap"):
            df_json = df.head(10000).to_json()
            # Dummy auto-mapping logic
            revenue = float(df[df.columns[-1]].sum())
            st.session_state["mapped_values"] = {
                "revenue": revenue, "cogs": 0.0, "opex": 0.0, "other_income": 0.0, "other_expenses": 0.0
            }
            st.success("Auto-map completed — check P&L Mapping tab to adjust/confirm.")
    else:
        st.info("Upload a file or use QuickBooks (simulated) to proceed.")

# ---------------------------
# Tab 2: P&L Mapping
# ---------------------------
with tab2:
    st.subheader("🧭 Map P&L → Revenue / COGS / OPEX / Other")
    if st.session_state.get("pl_df") is None:
        st.warning("No data found. Go to 'Data Import' first or manually enter P&L below.")
    else:
        df = st.session_state["pl_df"].copy()
        st.write("Auto-detect common columns (Account/Amount) or provide manual values.")
        if st.button("Auto-Map (cached)", key="tab2_btn_automap"):
            df_json = df.head(10000).to_json()
            revenue = float(df[df.columns[-1]].sum())
            st.session_state["mapped_values"] = {
                "revenue": revenue, "cogs": 0.0, "opex": 0.0, "other_income": 0.0, "other_expenses": 0.0
            }
            st.success("Auto-mapping complete (you can override the values).")
    mv = st.session_state.get("mapped_values", {})
    st.markdown("#### Manual / Override entries (edit and press Update)")
    with st.form("tab2_form_pnl_manual", clear_on_submit=False):
        revenue = st.number_input("Revenue (UGX)", min_value=0.0, value=float(mv.get("revenue", 0.0)), step=1000.0, format="%.2f", key="tab2_in_revenue")
        cogs = st.number_input("COGS (UGX)", min_value=0.0, value=float(mv.get("cogs", 0.0)), step=1000.0, format="%.2f", key="tab2_in_cogs")
        opex = st.number_input("Operating Expenses (UGX)", min_value=0.0, value=float(mv.get("opex", 0.0)), step=1000.0, format="%.2f", key="tab2_in_opex")
        other_income = st.number_input("Other Income (UGX)", min_value=0.0, value=float(mv.get("other_income", 0.0)), step=1000.0, format="%.2f", key="tab2_in_oi")
        other_expenses = st.number_input("Other Expenses (UGX)", min_value=0.0, value=float(mv.get("other_expenses", 0.0)), step=1000.0, format="%.2f", key="tab2_in_oe")
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

# ---------------------------
# Tab 3: Compute & Credits
# ---------------------------
with tab3:
    st.subheader("🧮 Compute Tax, Apply Credits & Exemptions")

    client_name = st.text_input("Client Name", value="Acme Ltd", key="tab3_client_name")
    tin = st.text_input("TIN (optional)", key="tab3_tin")

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

    # Addbacks (Disallowables)
    addbacks_labels = [
        "Depreciation", "Amortisation", "Redundancy", "Domestic/Private Expenditure",
        "Capital Gain", "Rental Income Loss", "Expenses Exceeding 50% of Rental Income",
        "Capital Nature Expenditure", "Recoverable Expenditure", "Income Tax Paid Abroad",
        "Capitalised Income", "Gift Cost not in Recipient Income", "Fines or Penalties",
        "Employee Retirement Contributions", "Life Insurance Premiums", "Pension Payments",
        "Alimony / Allowance", "Suppliers without TIN > UGX5M", "EFRIS Suppliers w/o e-invoices",
        "Debt Obligation Principal", "Interest on Capital Assets", "Interest on Fixed Capital",
        "Bad Debts Recovered", "General Provision for Bad Debts", "Entertainment Income",
        "Meal & Refreshment Expenses", "Charitable Donations to Non-Exempt Orgs",
        "Charitable Donations >5% Chargeable Income", "Legal Fees", "Legal Expenses - Capital Items",
        "Legal Expenses - New Trade Rights", "Legal Expenses - Breach of Law",
        "Cost of Breach of Contract - Capital Account", "Legal Expenses on Breach of Contract - Capital Account",
        "Legal Expenses on Loan Renewals - Non-commercial", "Bad Debts by Senior Employee/Management",
        "General Provisions Bad Debts", "Loss on Sale of Fixed Assets", "Loss on Other Capital Items",
        "Expenditure on Share Capital Increase", "Dividends Paid", "Provision for Bad Debts",
        "Increase in Provision for Bad Debts", "Debt Collection Expenses related to Capital Expenditure",
        "Foreign Currency Debt Gains", "Costs incidental to Capital Asset", "Non-Business Expenses",
        "Miscellaneous Staff Costs", "Staff Costs - Commuting", "First Time Work Permits",
        "Unrealised Foreign Exchange Losses", "Foreign Currency Debt Losses", "Education Expenditure",
        "Donations", "Decommissioning Expenditure by Licensee", "Telephone Costs (10%)",
        "Revaluation Loss", "Interest Expense on Treasury Bills", "Burial Expenses",
        "Subscription", "Interest on Directors Debit Balances", "Entertainment Expenses",
        "Gifts", "Dividends Paid (duplicate)", "Income Carried to Reserve Fund",
        "Impairment Losses on Loans and Advances", "Interest Expense on Treasury Bonds",
        "Staff Leave Provisions", "Increase in Gratuity", "Balancing Charge"
    ]
    with st.expander("Addbacks (Disallowable Expenses) — click to edit and save", expanded=False):
        with st.form("tab3_form_addbacks"):
            ab_values = {}
            for label in addbacks_labels:
                key = f"tab3_ab_{re.sub(r'[^a-z0-9]+', '_', label.lower())}"
                default_val = float(st.session_state.get("addbacks_values", {}).get(key, 0.0))
                ab_values[key] = st.number_input(label, min_value=0.0, value=default_val, format="%.2f", key=key + "_widget")
            addbacks_submit = st.form_submit_button("Save Addbacks", use_container_width=True)
            if addbacks_submit:
                if "addbacks_values" not in st.session_state:
                    st.session_state["addbacks_values"] = {}
                st.session_state["addbacks_values"].update(ab_values)
                st.success("Addbacks saved to session.")

    total_addbacks = sum(float(v) for v in st.session_state.get("addbacks_values", {}).values())
    adjusted_profit = pbit + total_addbacks
    st.markdown(f"### Adjusted Profit (PBIT + Addbacks): UGX {adjusted_profit:,.2f}")

    # Allowables (Deductions)
    allowables_labels = [
        "Wear & Tear", "Industrial Building Allowance", "Startup Costs",
        "Reverse VAT", "Listing Business with Uganda Stock Exchange",
        "Registration Fees, Accountant Fees, Legal Fees, Advertising, Training",
        "Expenses in Acquiring Intangible Asset", "Disposal of Intangible Asset",
        "Minor Capital Expenditure", "Revenue Expenditures - Repairs & Maintenance",
        "Expenditure on Scientific Research", "Expenditure on Training (Education)",
        "Charitable Donations to Exempt Organisations", "Charitable Donations Up to 5% Chargeable Income",
        "Expenditure on Farming", "Apportionment of Deductions", "Carry Forward Losses from Previous Period",
        "Carry Forward Losses Upto 50% after 7 Years", "Disposal of Trading Stock",
        "Foreign Currency Debt Loss", "Loss on Disposal of Asset", "Exclusion of Doctrine Mutuality",
        "Partnership Loss for Resident Partner", "Partnership Loss for Non-Resident Partner",
        "Expenditure or Loss by Trustee Beneficiary", "Expenditure or Loss by Beneficiary of Deceased Estate",
        "Limitation on Deduction for Petroleum Operations", "Decommission Costs & Expenditures - Petroleum",
        "Unrealised Gains", "Impairment of Asset", "Decrease in Provision for Bad Debts",
        "Bad Debts Written Off", "Staff Costs - Business Travel", "Private Employer Disability Tax",
        "Rental Income Expenditure & Losses", "Local Service Tax", "Interest Income on Treasury Bills",
        "Interest on Circulating Capital", "Interest Income on Treasury Bonds",
        "Specific Provisions for Bad Debts", "Revaluation Gains", "Rental Income",
        "Interest Income from Treasury Bills", "Interest Income from Treasury Bonds",
        "Legal Expenses on Breach of Contract to Revenue Account", "Legal Expenses on Maintenance of Capital Assets",
        "Legal Expenses on Existing Trade Rights", "Legal Expenses Incidental to Revenue Items",
        "Legal Expenses on Debt Collection - Trade Debts", "Closing Tax Written Down Value < UGX1M",
        "Intangible Assets", "Legal Expenses for Renewal of Loans", "Interest on Debt Obligation",
        "Interest on Debt Obligation by Group Member", "Gains & Losses on Disposal of Assets",
        "Balancing Allowance"
    ]
    with st.expander("Allowables (Deductions) — click to edit and save", expanded=False):
        with st.form("tab3_form_allowables"):
            al_values = {}
            for label in allowables_labels:
                key = f"tab3_al_{re.sub(r'[^a-z0-9]+', '_', label.lower())}"
                default_val = float(st.session_state.get("allowables_values", {}).get(key, 0.0))
                al_values[key] = st.number_input(label, min_value=0.0, value=default_val, format="%.2f", key=key + "_widget")
            allowables_submit = st.form_submit_button("Save Allowables", use_container_width=True)
            if allowables_submit:
                if "allowables_values" not in st.session_state:
                    st.session_state["allowables_values"] = {}
                st.session_state["allowables_values"].update(al_values)
                st.success("Allowables saved to session.")

    total_allowables = sum(float(v) for v in st.session_state.get("allowables_values", {}).values())
    chargeable_income = max(0.0, adjusted_profit - total_allowables)
    st.markdown(f"### Chargeable Income (after allowables): UGX {chargeable_income:,.2f}")

    st.markdown("### Credits, Capital Allowances & Rebates")
    col1, col2, col3 = st.columns(3)
    with col1:
        capital_allowances = st.number_input("Capital Allowances (UGX)", min_value=0.0, value=0.0, format="%.2f", key="tab3_in_capital_allowances")
        exemptions = st.number_input("Exemptions (UGX)", min_value=0.0, value=0.0, format="%.2f", key="tab3_in_exemptions")
    with col2:
        credits_wht = st.number_input("WHT Credits (UGX)", min_value=0.0, value=0.0, format="%.2f", key="tab3_in_wht")
        credits_foreign = st.number_input("Foreign Tax Credit (UGX)", min_value=0.0, value=0.0, format="%.2f", key="tab3_in_ftc")
    with col3:
        rebates = st.number_input("Rebates (UGX)", min_value=0.0, value=0.0, format="%.2f", key="tab3_in_rebates")
        provisional_tax_paid = st.number_input("Provisional Tax Paid (UGX)", min_value=0.0, value=0.0, format="%.2f", key="tab3_in_provisional")

    # Tax computation functions
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

    individual_brackets = [
        {"threshold": 0.0, "rate": 0.0, "fixed": 0.0},
        {"threshold": 2_820_000.0, "rate": 0.1, "fixed": 0.0},
        {"threshold": 4_020_000.0, "rate": 0.2, "fixed": 120_000.0},
        {"threshold": 4_920_000.0, "rate": 0.3, "fixed": 360_000.0},
        {"threshold": 10_000_000.0, "rate": 0.4, "fixed": 1_830_000.0}
    ]

    if st.button("Compute Tax Liability", key="tab3_btn_compute"):
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

        def save_history(row):
            conn = sqlite3.connect(HISTORY_DB)
            c = conn.cursor()
            keys = ','.join(row.keys())
            qmarks = ','.join(['?']*len(row))
            c.execute(f"INSERT INTO income_tax_history ({keys}) VALUES ({qmarks})", tuple(row.values()))
            conn.commit()
            conn.close()

        if st.button("💾 Save Computation to History (DB)", key="tab3_btn_save_history"):
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
            st.success("Saved to history.")

# ---------------------------
# Tab 4: Dashboard
# ---------------------------
with tab4:
    st.subheader("📊 Multi-Year History Dashboard")
    def load_history_cached(client_filter: str = "") -> pd.DataFrame:
        try:
            with sqlite3.connect(HISTORY_DB) as conn:
                df = pd.read_sql_query(
                    "SELECT * FROM income_tax_history ORDER BY year DESC, created_at DESC", conn
                )
            if client_filter and not df.empty:
                df = df[df["client_name"].str.contains(client_filter, case=False, na=False)]
            return df
        except Exception as e:
            st.error(f"Failed to load history: {e}")
            return pd.DataFrame()

    client_filter = st.text_input("Filter by client name (optional)", "", key="dashboard_client_filter")
    hist = load_history_cached(client_filter)
    if hist.empty:
        st.info("No saved history yet.")
    else:
        st.write("Showing latest 200 records (use filter to narrow).")
        st.dataframe(hist.head(200), use_container_width=True)
        st.markdown("#### Net Tax by Year")
        pivot = hist.groupby(["year"])["net_tax_payable"].sum().reset_index()
        if not pivot.empty:
            st.line_chart(
                pivot.rename(columns={"net_tax_payable": "Net Tax Payable"}).set_index("year")
            )
        st.markdown("#### Taxable Income vs Gross Tax (latest 30)")
        chart_df = hist.head(30).set_index("created_at")[["taxable_income", "gross_tax"]]
        st.bar_chart(chart_df)

# ---------------------------
# Tab 5: Export & URA Forms
# ---------------------------
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
    TIN_input = st.text_input("TIN (required)", value=last.get("TIN", ""), key="ura_tin_input")
    if form_code == "DT-2001":
        taxpayer_name = st.text_input("Taxpayer Name", value=suggested_client, key="ura_taxpayer_name")
        business_income = st.number_input("Business Income (UGX)", min_value=0.0, value=suggested_taxable, format="%.2f", key="ura_biz_income")
        allowable_deductions = st.number_input("Allowable Deductions (UGX)", min_value=0.0, value=float(last.get("total_allowables", 0.0)), format="%.2f", key="ura_allowable_deductions")
        capital_allowances_f = st.number_input("Capital Allowances (UGX)", min_value=0.0, value=float(last.get("capital_allowances", 0.0)), format="%.2f", key="ura_cap_allowances")
        exemptions_f = st.number_input("Exemptions (UGX)", min_value=0.0, value=float(last.get("exemptions", 0.0)), format="%.2f", key="ura_exemptions")
        gross_tax_f = st.number_input("Gross Tax (UGX)", min_value=0.0, value=suggested_gross, format="%.2f", key="ura_gross_tax")
        wht_f = st.number_input("WHT Credits (UGX)", min_value=0.0, value=float(last.get("credits_wht", 0.0)), format="%.2f", key="ura_wht")
        foreign_f = st.number_input("Foreign Tax Credit (UGX)", min_value=0.0, value=float(last.get("credits_foreign", 0.0)), format="%.2f", key="ura_ftc")
        rebates_f = st.number_input("Rebates (UGX)", min_value=0.0, value=float(last.get("rebates", 0.0)), format="%.2f", key="ura_rebates")
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
            "Net Tax Payable (UGX)": max(0.0, gross_tax_f - wht_f - foreign_f - rebates_f),
        }
    else:
        entity_name = st.text_input("Entity Name", value=suggested_client, key="ura_entity_name")
        gross_turnover = st.number_input("Gross Turnover (UGX)", min_value=0.0, value=float(last.get("revenue", 0.0)), format="%.2f", key="ura_gturnover")
        cogs_f = st.number_input("COGS (UGX)", min_value=0.0, value=float(last.get("cogs", 0.0)), format="%.2f", key="ura_cogs")
        opex_f = st.number_input("Operating Expenses (UGX)", min_value=0.0, value=float(last.get("opex", 0.0)), format="%.2f", key="ura_opex")
        other_income_f = st.number_input("Other Income (UGX)", min_value=0.0, value=float(last.get("other_income", 0.0)), format="%.2f", key="ura_oincome")
        other_expenses_f = st.number_input("Other Expenses (UGX)", min_value=0.0, value=float(last.get("other_expenses", 0.0)), format="%.2f", key="ura_oexpense")
        capital_allowances_f = st.number_input("Capital Allowances (UGX)", min_value=0.0, value=float(last.get("capital_allowances", 0.0)), format="%.2f", key="ura_cap_allowances_c")
        exemptions_f = st.number_input("Exemptions (UGX)", min_value=0.0, value=float(last.get("exemptions", 0.0)), format="%.2f", key="ura_exemptions_c")
        gross_tax_f = st.number_input("Gross Tax (UGX)", min_value=0.0, value=suggested_gross, format="%.2f", key="ura_gross_tax_c")
        wht_f = st.number_input("WHT Credits (UGX)", min_value=0.0, value=float(last.get("credits_wht", 0.0)), format="%.2f", key="ura_wht_c")
        foreign_f = st.number_input("Foreign Tax Credit (UGX)", min_value=0.0, value=float(last.get("credits_foreign", 0.0)), format="%.2f", key="ura_ftc_c")
        rebates_f = st.number_input("Rebates (UGX)", min_value=0.0, value=float(last.get("rebates", 0.0)), format="%.2f", key="ura_rebates_c")
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
            "Net Tax Payable (UGX)": max(0.0, gross_tax_f - wht_f - foreign_f - rebates_f),
        }
    def validate_and_build_return(form_code, payload):
        for k, v in payload.items():
            if v is None or (isinstance(v, str) and not v.strip()):
                raise ValueError(f"Missing value for {k}")
        df = pd.DataFrame([payload])
        return df
    if st.button("✅ Validate & Build CSV / Excel", key="ura_btn_build"):
        try:
            df_return = validate_and_build_return(form_code, payload)
            st.success("Validation passed. Download your URA return below.")
            csv_bytes = df_return.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download URA Return CSV",
                data=csv_bytes,
                file_name=f"{form_code}_{payload.get('Year')}_{payload.get('TIN','')}.csv",
                mime="text/csv",
                key="ura_dl_csv"
            )
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                df_return.to_excel(writer, index=False, sheet_name=form_code)
            st.download_button(
                label="📥 Download URA Return Excel",
                data=buffer.getvalue(),
                file_name=f"{form_code}_{payload.get('Year')}_{payload.get('TIN','')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="ura_dl_xlsx"
            )
            st.dataframe(df_return, use_container_width=True)
        except Exception as e:
            st.error(f"Validation failed: {e}")

# ---------------------------
# Tab 6: Audit & Controls
# ---------------------------
with tab6:
    st.subheader("🔎 Audit & Control Accounts")
    st.markdown("""
        Upload a **Trial Balance** (TB). This tool will:
        1) Check TB integrity (Debits = Credits),
        2) Match & test sign expectations for control accounts,
        3) Reconcile TB-derived P&L vs your mapped P&L values.
    """)
    tb_file = st.file_uploader("Upload Trial Balance (CSV/Excel)", type=["csv", "xlsx"], key="audit_tb_upload")
    materiality = st.sidebar.number_input(
        "Materiality Threshold (UGX)", min_value=0.0, value=1000000.0, step=100000.0, key="audit_materiality_threshold"
    )
    control_accounts = {
        "Cash": "Debit",
        "Bank": "Debit",
        "Accounts Receivable": "Debit",
        "Accounts Payable": "Credit",
        "Revenue": "Credit",
        "Expenses": "Debit"
    }
    user_control_map = {}
    for i, (acc, expected_sign) in enumerate(control_accounts.items()):
        user_choice = st.sidebar.selectbox(
            f"{acc} Expected Sign", ["Debit", "Credit"],
            index=0 if expected_sign == "Debit" else 1,
            key=f"audit_ctrl_{acc}_{i}"
        )
        user_control_map[acc] = user_choice
    if tb_file:
        try:
            if tb_file.name.endswith(".csv"):
                tb_df = pd.read_csv(tb_file)
            else:
                tb_df = pd.read_excel(tb_file)
            st.write("✅ Trial Balance Preview:")
            st.dataframe(tb_df.head())
            total_debits = tb_df["Debit"].sum() if "Debit" in tb_df.columns else 0
            total_credits = tb_df["Credit"].sum() if "Credit" in tb_df.columns else 0
            if np.isclose(total_debits, total_credits):
                st.success(f"Trial Balance is balanced. Debits = {total_debits}, Credits = {total_credits}")
            else:
                st.error(f"Trial Balance is NOT balanced! Debits = {total_debits}, Credits = {total_credits}")
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
            st.markdown("### P&L Reconciliation")
            tb_pl_total = tb_df.loc[tb_df["Account"].isin(["Revenue", "Expenses"]), "Debit"].sum() - tb_df.loc[tb_df["Account"].isin(["Revenue", "Expenses"]), "Credit"].sum()
            if abs(tb_pl_total) <= materiality:
                st.success(f"P&L within materiality ({tb_pl_total} UGX)")
            else:
                st.warning(f"P&L outside materiality ({tb_pl_total} UGX)")
        except Exception as e:
            st.error(f"Error processing TB: {e}")

# ---------------------------
# Tab 7: VAT
# ---------------------------
with tab7:
    st.header("VAT Return")
    st.subheader("Monthly VAT Computation & Return")
    st.markdown("Enter your VAT sales and purchases for the period below:")

    vat_sales = st.number_input("Total VATable Sales (UGX)", min_value=0.0, value=0.0, step=1000.0, format="%.2f", key="vat_sales_tab7")
    vat_output = st.number_input("Output VAT Collected (UGX)", min_value=0.0, value=0.0, step=1000.0, format="%.2f", key="vat_output_tab7")
    vat_purchases = st.number_input("Total VATable Purchases (UGX)", min_value=0.0, value=0.0, step=1000.0, format="%.2f", key="vat_purchases_tab7")
    vat_input = st.number_input("Input VAT Paid (UGX)", min_value=0.0, value=0.0, step=1000.0, format="%.2f", key="vat_input_tab7")
    vat_adjustments = st.number_input("Adjustments (UGX)", min_value=0.0, value=0.0, step=1000.0, format="%.2f", key="vat_adjustments_tab7")

    net_vat_payable = max(0.0, vat_output - vat_input + vat_adjustments)
    st.metric("Net VAT Payable", f"UGX {net_vat_payable:,.2f}")

    if st.button("Download VAT Return CSV", key="vat_dl_btn_tab7"):
        import pandas as pd
        vat_df = pd.DataFrame([{
            "VATable Sales": vat_sales,
            "Output VAT": vat_output,
            "VATable Purchases": vat_purchases,
            "Input VAT": vat_input,
            "Adjustments": vat_adjustments,
            "Net VAT Payable": net_vat_payable
        }])
        st.download_button(
            label="Download VAT Return CSV",
            data=vat_df.to_csv(index=False).encode("utf-8"),
            file_name="VAT_Return.csv",
            mime="text/csv",
            key="vat_dl_tab7"
        )

# ---------------------------
# Tab 8: Presumptive Tax
# ---------------------------
with tab8:
    st.header("Presumptive Tax")
    st.subheader("Presumptive Tax Calculator (for small businesses)")
    st.markdown("Enter your annual gross turnover below:")

    gross_turnover = st.number_input("Annual Gross Turnover (UGX)", min_value=0.0, value=0.0, step=1000.0, format="%.2f", key="presumptive_turnover_tab8")
    has_book_of_accounts = st.checkbox("Do you keep books of accounts?", value=True, key="presumptive_books_tab8")
    tax_due = 0.0

    if gross_turnover <= 10_000_000:
        tax_due = 0.0
        st.info("No presumptive tax due (turnover ≤ UGX 10M).")
    elif gross_turnover <= 30_000_000:
        tax_due = 80_000.0
    elif gross_turnover <= 50_000_000:
        tax_due = 200_000.0
    elif gross_turnover <= 80_000_000:
        tax_due = 400_000.0
    elif gross_turnover <= 150_000_000:
        tax_due = 900_000.0
    elif gross_turnover <= 350_000_000:
        tax_due = 2_200_000.0
    elif gross_turnover <= 500_000_000:
        tax_due = 2_700_000.0
    else:
        st.warning("Turnover exceeds presumptive tax threshold (UGX 500M). Use normal income tax.")

    if has_book_of_accounts and gross_turnover > 50_000_000 and gross_turnover <= 150_000_000:
        tax_due = gross_turnover * 0.015

    st.metric("Presumptive Tax Due", f"UGX {tax_due:,.2f}")

    if st.button("Download Presumptive Tax CSV", key="presumptive_dl_btn_tab8"):
        import pandas as pd
        presumptive_df = pd.DataFrame([{
            "Gross Turnover": gross_turnover,
            "Keeps Books of Accounts": has_book_of_accounts,
            "Presumptive Tax Due": tax_due
        }])
        st.download_button(
            label="Download Presumptive Tax CSV",
            data=presumptive_df.to_csv(index=False).encode("utf-8"),
            file_name="Presumptive_Tax.csv",
            mime="text/csv",
            key="presumptive_dl_tab8"
        )

# ---------------------------
# Tab 9: Payroll
# ---------------------------
with tab9:
    import pandas as pd
    from io import BytesIO
    st.header("Payroll for the Month of XXXX")
    num_employees = st.number_input("Number of Employees", min_value=1, value=3, key="num_employees_tab9")
    columns = ["Name", "TIN Number", "Basic Salary", "Allowances", "Bonus", "Gross Pay",
               "LST Chargeable Income", "PAYE", "NSSF 5%", "LST", "Total Deduction", "Net Pay", "Take Home Pay"]
    payroll_df = pd.DataFrame(columns=columns)
    def calculate_lst_chargeable(gross_pay):
        if gross_pay > 10000000:
            return 25000 + 0.3 * (gross_pay - 410000) + 0.1 * (gross_pay - 10000000)
        elif gross_pay > 410000:
            return 25000 + 0.3 * (gross_pay - 410000)
        elif gross_pay > 335000:
            return 10000 + 0.2 * (gross_pay - 335000)
        elif gross_pay > 235000:
            return 0.1 * (gross_pay - 235000)
        else:
            return 0
    def calculate_paye(lst_chargeable):
        return lst_chargeable
    def calculate_nssf(basic_salary):
        return 0.05 * basic_salary
    def calculate_total_deductions(paye, nssf, lst):
        return paye + nssf + lst
    for i in range(num_employees):
        st.subheader(f"Employee {i+1}")
        name = st.text_input(f"Name {i+1}", key=f"name_tab9_{i}")
        tin = st.text_input(f"TIN Number {i+1}", key=f"tin_tab9_{i}")
        basic_salary = st.number_input(f"Basic Salary {i+1}", min_value=0, key=f"basic_tab9_{i}")
        allowances = st.number_input(f"Allowances {i+1}", min_value=0, key=f"allow_tab9_{i}")
        bonus = st.number_input(f"Bonus {i+1}", min_value=0, key=f"bonus_tab9_{i}")
        gross_pay = basic_salary + allowances + bonus
        lst_chargeable = calculate_lst_chargeable(gross_pay)
        paye = calculate_paye(lst_chargeable)
        nssf = calculate_nssf(basic_salary)
        lst = lst_chargeable
        total_deduction = calculate_total_deductions(paye, nssf, lst)
        net_pay = gross_pay - total_deduction
        take_home = net_pay
        payroll_df = pd.concat([payroll_df, pd.DataFrame([{
            "Name": name,
            "TIN Number": tin,
            "Basic Salary": basic_salary,
            "Allowances": allowances,
            "Bonus": bonus,
            "Gross Pay": gross_pay,
            "LST Chargeable Income": lst_chargeable,
            "PAYE": paye,
            "NSSF 5%": nssf,
            "LST": lst,
            "Total Deduction": total_deduction,
            "Net Pay": net_pay,
            "Take Home Pay": take_home
        }])], ignore_index=True)
    st.subheader("Payroll Summary")
    st.dataframe(payroll_df.style.format({
        "Basic Salary": "{:,.2f}",
        "Allowances": "{:,.2f}",
        "Bonus": "{:,.2f}",
        "Gross Pay": "{:,.2f}",
        "LST Chargeable Income": "{:,.2f}",
        "PAYE": "{:,.2f}",
        "NSSF 5%": "{:,.2f}",
        "LST": "{:,.2f}",
        "Total Deduction": "{:,.2f}",
        "Net Pay": "{:,.2f}",
        "Take Home Pay": "{:,.2f}"
    }))
    def to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Payroll')
            workbook  = writer.book
            worksheet = writer.sheets['Payroll']
            money_fmt = workbook.add_format({'num_format': '#,##0.00'})
            for col_num, value in enumerate(df.columns):
                if col_num >= 2:
                    worksheet.set_column(col_num, col_num, 15, money_fmt)
                else:
                    worksheet.set_column(col_num, col_num, 20)
        processed_data = output.getvalue()
        return processed_data
    excel_data = to_excel(payroll_df)
    st.download_button(
        label="Download Payroll as Excel",
        data=excel_data,
        file_name="Payroll_XYZ_Co.xlsx",
        mime="application/vnd.ms-excel",
        key="payroll_dl_tab9"
    )

# ---------------------------
# Tab 10: Withholding Tax (WHT)
# ---------------------------
with tab10:
    import pandas as pd
    from io import BytesIO
    st.header("Withholding Tax (WHT) - XYZ Co Ltd")
    st.header("EAST Trainers TIN: 10000xxxx")
    period = st.text_input("Period", value="MAY 2025", key="wht_period_tab10")
    st.subheader("International Payments")
    intl_columns = ["Inv. Date", "Invoice No.", "Country", "TIN", "Supplier", "Description",
                    "Currency", "Amount excl. WHT", "WHT Rate (%)", "WHT Amount", "URA Rate", "Amount Ushs", "WHT Ushs"]
    intl_df = pd.DataFrame(columns=intl_columns)
    num_intl = st.number_input("Number of International Payments", min_value=1, value=6, key="num_intl_tab10")
    for i in range(num_intl):
        st.markdown(f"**International Payment {i+1}**")
        inv_date = st.date_input(f"Invoice Date {i+1}", key=f"intl_date_tab10_{i}")
        invoice_no = st.text_input(f"Invoice No {i+1}", key=f"intl_inv_tab10_{i}")
        country = st.text_input(f"Country {i+1}", key=f"intl_country_tab10_{i}")
        tin = st.text_input(f"TIN {i+1}", key=f"intl_tin_tab10_{i}")
        supplier = st.text_input(f"Supplier {i+1}", key=f"intl_supplier_tab10_{i}")
        description = st.text_input(f"Description {i+1}", key=f"intl_desc_tab10_{i}")
        currency = st.selectbox(f"Currency {i+1}", ["USD", "EUR"], key=f"intl_cur_tab10_{i}")
        amount = st.number_input(f"Amount excl. WHT {i+1}", min_value=0.0, key=f"intl_amount_tab10_{i}")
        wht_rate = st.number_input(f"WHT Rate % {i+1}", min_value=0.0, value=15.0, key=f"intl_wht_rate_tab10_{i}")
        ura_rate = st.number_input(f"URA Rate {i+1}", value=3651 if currency=="USD" else 4142, key=f"intl_ura_rate_tab10_{i}")
        wht_amount = amount * (wht_rate / 100)
        amount_ushs = amount * ura_rate
        wht_ushs = wht_amount * ura_rate
        intl_df = pd.concat([intl_df, pd.DataFrame([{
            "Inv. Date": inv_date,
            "Invoice No.": invoice_no,
            "Country": country,
            "TIN": tin,
            "Supplier": supplier,
            "Description": description,
            "Currency": currency,
            "Amount excl. WHT": amount,
            "WHT Rate (%)": wht_rate,
            "WHT Amount": wht_amount,
            "URA Rate": ura_rate,
            "Amount Ushs": amount_ushs,
            "WHT Ushs": wht_ushs
        }])], ignore_index=True)
    st.dataframe(intl_df.style.format({
        "Amount excl. WHT": "{:,.2f}",
        "WHT Amount": "{:,.2f}",
        "Amount Ushs": "{:,.2f}",
        "WHT Ushs": "{:,.2f}"
    }))
    st.subheader("Local Payments")
    local_columns = ["Inv. Date", "Invoice No.", "TIN", "Supplier", "Description",
                     "Currency", "Amount excl. WHT", "WHT Rate (%)", "WHT Amount", "URA Rate", "Amount Ushs", "WHT Ushs"]
    local_df = pd.DataFrame(columns=local_columns)
    num_local = st.number_input("Number of Local Payments", min_value=1, value=4, key="num_local_tab10")
    for i in range(num_local):
        st.markdown(f"**Local Payment {i+1}**")
        inv_date = st.date_input(f"Invoice Date {i+1}", key=f"loc_date_tab10_{i}")
        invoice_no = st.text_input(f"Invoice No {i+1}", key=f"loc_inv_tab10_{i}")
        tin = st.text_input(f"TIN {i+1}", key=f"loc_tin_tab10_{i}")
        supplier = st.text_input(f"Supplier {i+1}", key=f"loc_supplier_tab10_{i}")
        description = st.text_input(f"Description {i+1}", key=f"loc_desc_tab10_{i}")
        currency = st.selectbox(f"Currency {i+1}", ["UGX", "USD", "EUR"], key=f"loc_cur_tab10_{i}")
        amount = st.number_input(f"Amount excl. WHT {i+1}", min_value=0.0, key=f"loc_amount_tab10_{i}")
        wht_rate = st.number_input(f"WHT Rate % {i+1}", min_value=0.0, value=6.0, key=f"loc_wht_rate_tab10_{i}")
        ura_rate = st.number_input(f"URA Rate {i+1}", value=1 if currency=="UGX" else 3651, key=f"loc_ura_rate_tab10_{i}")
        wht_amount = amount * (wht_rate / 100)
        amount_ushs = amount * ura_rate
        wht_ushs = wht_amount * ura_rate
        local_df = pd.concat([local_df, pd.DataFrame([{
            "Inv. Date": inv_date,
            "Invoice No.": invoice_no,
            "TIN": tin,
            "Supplier": supplier,
            "Description": description,
            "Currency": currency,
            "Amount excl. WHT": amount,
            "WHT Rate (%)": wht_rate,
            "WHT Amount": wht_amount,
            "URA Rate": ura_rate,
            "Amount Ushs": amount_ushs,
            "WHT Ushs": wht_ushs
        }])], ignore_index=True)
    st.dataframe(local_df.style.format({
        "Amount excl. WHT": "{:,.2f}",
        "WHT Amount": "{:,.2f}",
        "Amount Ushs": "{:,.2f}",
        "WHT Ushs": "{:,.2f}"
    }))
    total_wht_payable = intl_df["WHT Ushs"].sum() + local_df["WHT Ushs"].sum()
    st.subheader(f"TOTAL WHT PAYABLE: {total_wht_payable:,.0f} UGX")
    def wht_to_excel(intl_df, local_df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            intl_df.to_excel(writer, index=False, sheet_name='International Payments')
            local_df.to_excel(writer, index=False, sheet_name='Local Payments')
            workbook  = writer.book
            money_fmt = workbook.add_format({'num_format': '#,##0.00'})
            for sheet_name in ['International Payments', 'Local Payments']:
                worksheet = writer.sheets[sheet_name]
                df = intl_df if sheet_name == 'International Payments' else local_df
                for col_num, value in enumerate(df.columns):
                    if col_num >= 7:
                        worksheet.set_column(col_num, col_num, 15, money_fmt)
                    else:
                        worksheet.set_column(col_num, col_num, 20)
        return output.getvalue()
    excel_data = wht_to_excel(intl_df, local_df)
    st.download_button(
        label="Download WHT as Excel",
        data=excel_data,
        file_name="WHT_XYZ_Co.xlsx",
        mime="application/vnd.ms-excel",
        key="wht_dl_tab10"
    )

# ---------------------------
# Tab 11: Transfer Pricing
# ---------------------------
with tab11:
    st.header("Transfer Pricing")
    st.subheader("Transfer Pricing Disclosure & Documentation")
    st.markdown("""
        Use this section to record and summarize your related party transactions for transfer pricing compliance.
        Fill in details for each transaction below. You can download your disclosure as an Excel file for URA submission.
    """)

    num_tp = st.number_input("Number of Related Party Transactions", min_value=1, value=2, key="tp_num_tab11")
    tp_columns = [
        "Date", "Related Party Name", "Related Party TIN", "Country", "Transaction Type",
        "Description", "Amount (UGX)", "Currency", "Method Used", "Comparable Uncontrolled Price"
    ]
    tp_df = pd.DataFrame(columns=tp_columns)
    for i in range(num_tp):
        st.markdown(f"**Transaction {i+1}**")
        date = st.date_input(f"Date {i+1}", key=f"tp_date_tab11_{i}")
        party_name = st.text_input(f"Related Party Name {i+1}", key=f"tp_party_name_tab11_{i}")
        party_tin = st.text_input(f"Related Party TIN {i+1}", key=f"tp_party_tin_tab11_{i}")
        country = st.text_input(f"Country {i+1}", key=f"tp_country_tab11_{i}")
        tx_type = st.selectbox(f"Transaction Type {i+1}", ["Sale", "Purchase", "Service", "Loan", "Royalty", "Other"], key=f"tp_tx_type_tab11_{i}")
        description = st.text_input(f"Description {i+1}", key=f"tp_desc_tab11_{i}")
        amount = st.number_input(f"Amount (UGX) {i+1}", min_value=0.0, key=f"tp_amount_tab11_{i}")
        currency = st.selectbox(f"Currency {i+1}", ["UGX", "USD", "EUR"], key=f"tp_currency_tab11_{i}")
        method = st.selectbox(f"Method Used {i+1}", ["CUP", "Resale Price", "Cost Plus", "TNMM", "Profit Split", "Other"], key=f"tp_method_tab11_{i}")
        cup = st.number_input(f"Comparable Uncontrolled Price {i+1}", min_value=0.0, key=f"tp_cup_tab11_{i}")
        tp_df = pd.concat([tp_df, pd.DataFrame([{
            "Date": date,
            "Related Party Name": party_name,
            "Related Party TIN": party_tin,
            "Country": country,
            "Transaction Type": tx_type,
            "Description": description,
            "Amount (UGX)": amount,
            "Currency": currency,
            "Method Used": method,
            "Comparable Uncontrolled Price": cup
        }])], ignore_index=True)

    st.subheader("Transfer Pricing Disclosure Summary")
    st.dataframe(tp_df.style.format({
        "Amount (UGX)": "{:,.2f}",
        "Comparable Uncontrolled Price": "{:,.2f}"
    }))

    from io import BytesIO
    def tp_to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Transfer Pricing')
            workbook  = writer.book
            worksheet = writer.sheets['Transfer Pricing']
            money_fmt = workbook.add_format({'num_format': '#,##0.00'})
            for col_num, value in enumerate(df.columns):
                if value in ["Amount (UGX)", "Comparable Uncontrolled Price"]:
                    worksheet.set_column(col_num, col_num, 20, money_fmt)
                else:
                    worksheet.set_column(col_num, col_num, 20)
        return output.getvalue()

    excel_data = tp_to_excel(tp_df)
    st.download_button(
        label="Download Transfer Pricing Disclosure as Excel",
        data=excel_data,
        file_name="Transfer_Pricing_Disclosure.xlsx",
        mime="application/vnd.ms-excel",
        key="tp_dl_tab11"
    )

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size:12px; color:gray;'>TaxIntellilytics &copy; 2025 | Developed by Walter Hillary Kaijamahe</div>",
    unsafe_allow_html=True
)

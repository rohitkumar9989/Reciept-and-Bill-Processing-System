import streamlit as st
import sqlite3
import pandas as pd
import easyocr
from PIL import Image
from pdf2image import convert_from_path
import numpy as np
from collections import Counter, deque
import re
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from datetime import date, datetime
from pydantic import BaseModel, validator, ValidationError
from typing import Optional

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

VENDOR_CATEGORY = {
    "Amazon": "Shopping",
    "Walmart": "Groceries",
    "Starbucks": "Food & Beverage",
    "Big Bazaar": "Groceries",
    "Dominos": "Food & Beverage",
    "Apple": "Electronics",
    "Samsung": "Electronics",
    "Flipkart": "Shopping",
    "Uber": "Transport",
    "Reliance": "Groceries",
}

def infer_category(vendor):
    """
    Infers expense category from a vendor string using a vendor-category mapping.
    """
    for key, value in VENDOR_CATEGORY.items():
        if key.lower() in vendor.lower():
            return value
    return "Other"

conn = sqlite3.connect('receipts.db', check_same_thread=False)
c = conn.cursor()
c.execute("""
    CREATE TABLE IF NOT EXISTS receipts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        item TEXT,
        amount REAL,
        category TEXT
    )
""")
try:
    c.execute("ALTER TABLE receipts ADD COLUMN category TEXT")
except sqlite3.OperationalError:
    pass
conn.commit()

reader = easyocr.Reader(['en'])

class ReceiptData(BaseModel):
    """
    Structured data model for receipts with validation.
    """
    vendor: str
    date: str
    amount: float
    category: Optional[str] = None

    @validator("date")
    def validate_date(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except Exception:
            raise ValueError("Date must be in format YYYY-MM-DD")
        return v

    @validator("amount")
    def check_positive(cls, v):
        if v is None:
            raise ValueError("Amount cannot be empty")
        if v < 0:
            raise ValueError("Amount must be non-negative")
        return v

def extract_text(file):
    """
    Extracts plain text from filename using OCR for images/PDF and utf-8 for text files.
    """
    if file.name.endswith(".pdf"):
        images = convert_from_path(file.name)
        text = ""
        for img in images:
            img_np = np.array(img)
            result = reader.readtext(img_np, detail=0, paragraph=True)
            text += " ".join(result) + " "
        return text
    elif file.name.endswith((".jpg", ".png", ".jpeg")):
        img = Image.open(file)
        img_np = np.array(img)
        result = reader.readtext(img_np, detail=0, paragraph=True)
        return " ".join(result)
    elif file.name.endswith(".txt"):
        return str(file.read(), 'utf-8')
    else:
        return ""

def parse_receipt_fields(text):
    """
    Extracts and validates vendor, date, amount, and category fields from the raw receipt text.
    Returns a ReceiptData object or an error string.
    """
    vendor = None
    date_str = None
    amount = None

    m_date = re.search(r'(\d{4}[-/]\d{2}[-/]\d{2})', text)
    if m_date:
        date_str = m_date.group(1).replace('/', '-')

    m_amount = re.search(r'(?:Total\s*[:\-]?\s*\$?|Amount\s*[:\-]?\s*\$?|Rs\.?\s*|INR\s*)?(\d+[.,]?\d*)', text, flags=re.IGNORECASE)
    if m_amount:
        amount = float(m_amount.group(1).replace(",", ""))

    first_line = text.splitlines()[0]
    vendor = first_line.strip() if first_line else "Unknown"
    category = infer_category(vendor)

    try:
        receipt = ReceiptData(vendor=vendor, date=date_str, amount=amount, category=category)
        return receipt, None
    except ValidationError as e:
        return None, str(e)

def aggregate_word_clusters(words):
    """
    Aggregates a sequence of words into item name clusters using POS tagging.
    """
    if not words:
        return words
    try:
        tagged = pos_tag(words)
    except:
        return words
    clusters, current_cluster = [], []
    for i, (word, pos) in enumerate(tagged):
        if not current_cluster:
            current_cluster.append(word)
        else:
            prev_pos = tagged[i-1][1]
            should_cluster = (
                (prev_pos.startswith('JJ') and pos.startswith('NN')) or
                (prev_pos == 'CD' and pos.startswith('NN')) or
                (prev_pos.startswith('NN') and pos.startswith('NN')) or
                (prev_pos.startswith('NNP') and pos.startswith('NNP'))
            )
            if should_cluster:
                current_cluster.append(word)
            else:
                if current_cluster:
                    clusters.append(" ".join(current_cluster))
                current_cluster = [word]
    if current_cluster:
        clusters.append(" ".join(current_cluster))
    return clusters

def is_number(s):
    """
    Checks if a string is a number, robust to common OCR issues.
    """
    s = s.replace(",", "").replace("E", "").replace("e", "").replace("O", "0")
    try:
        float(s)
        return True
    except:
        return False

def parse_items_prices(text):
    """
    Extracts item-price mappings from raw receipt text using word clustering and positional heuristics.
    """
    tokens = deque(text.split())
    raw_items, prices = [], []
    while tokens:
        token = tokens[0]
        if is_number(token):
            break
        else:
            raw_items.append(tokens.popleft())
    while tokens:
        token = tokens.popleft()
        if is_number(token):
            prices.append(float(token.replace(",", "").replace("E", "").replace("O", "0")))
    clustered_items = aggregate_word_clusters(raw_items)
    result = {}
    for i, item in enumerate(clustered_items):
        price = prices[i] if i < len(prices) else 0.0
        result[item] = price
    return result

def linear_search(data, keyword):
    """
    Searches for keyword in item name fields of uploaded receipts.
    """
    result = []
    for record in data:
        if keyword.lower() in record['item'].lower():
            result.append(record)
    return result

def quicksort(arr, key):
    """
    Sorts a list of dictionaries by the given key using quicksort algorithm.
    """
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = [x for x in arr[1:] if x[key] <= pivot[key]]
    right = [x for x in arr[1:] if x[key] > pivot[key]]
    return quicksort(left, key) + [pivot] + quicksort(right, key)

def compute_mean(data):
    """
    Returns the mean of the input list.
    """
    if len(data) == 0: return 0
    return sum(data) / len(data)

def compute_median(data):
    """
    Returns the median value of the input list.
    """
    if len(data) == 0: return 0
    s = sorted(data)
    n = len(s)
    if n % 2 == 1:
        return s[n//2]
    else:
        return (s[n//2 - 1] + s[n//2]) / 2

def compute_mode(data):
    """
    Returns the most common (mode) value in the input list.
    """
    if len(data) == 0: return 0
    c = Counter(data)
    return c.most_common(1)[0][0]

def text_reader(file_obj):
    """
    Reads text file uploads; expects each line as 'item price'.
    """
    content = file_obj.read().decode("utf-8")
    lines = content.splitlines()
    pairs = {}
    for i in range(len(lines)):
        clean_line = re.sub(r'[^\w\s]', '', lines[i])
        clean_line = clean_line.split(" ")
        product = clean_line[0]
        price = float(clean_line[-1])
        pairs[product] = pairs.get(product, 0) + price
    return pairs

st.sidebar.title("Receipt Processor")
page = st.sidebar.radio("Menu", ["Upload", "Database", "Insights"])

if page == "Upload":
    st.title("Upload Receipt")
    f = st.file_uploader("Upload file (.jpg, .png, .pdf, .txt)", type=['jpg', 'png', 'pdf', 'txt'])
    if f:
        if f.name.lower().endswith(".txt"):
            pairs = text_reader(f)
            extracted_text = '\n'.join([f"{k} {v}" for k, v in pairs.items()])
        else:
            extracted_text = extract_text(f)
            pairs = parse_items_prices(extracted_text)

        receipt_struct, val_error = parse_receipt_fields(extracted_text)
        st.markdown("### Receipt Summary")
        if receipt_struct:
            st.info(f"Vendor: {receipt_struct.vendor}")
            st.info(f"Date: {receipt_struct.date}")
            st.info(f"Amount: {receipt_struct.amount}")
            st.info(f"Category: {receipt_struct.category}")
            receipt_category = receipt_struct.category
        elif val_error:
            st.error(f"Parsing/validation error: {val_error}")
            receipt_category = "Other"
        else:
            receipt_category = "Other"

        st.session_state.edited_pairs = pairs.copy()
        if "edited_pairs" not in st.session_state:
            st.session_state.edited_pairs = pairs.copy()
        df_display = pd.DataFrame([
            {
                "No.": idx+1,
                "Item Name": item,
                "Price": price,
                "Category": infer_category(item)
            }
            for idx, (item, price) in enumerate(st.session_state.edited_pairs.items())
        ])
        st.subheader("Clustered items & prices identified")
        st.table(df_display)
        st.markdown("### Edit a single line")
        labels = [f"{i+1} · {name}" for i, name in enumerate(st.session_state.edited_pairs)]
        label_to_item = dict(zip(labels, st.session_state.edited_pairs))
        selected = st.selectbox("Select item to edit", labels)
        orig_name = label_to_item[selected]
        orig_price = st.session_state.edited_pairs[orig_name]
        new_name = st.text_input("Product name", value=orig_name)
        new_price = st.number_input("Price", value=orig_price, min_value=0.0)
        new_cat = st.text_input("Category", value=infer_category(orig_name))
        if st.button("Update this row"):
            st.session_state.edited_pairs.pop(orig_name)
            st.session_state.edited_pairs[new_name] = new_price
            st.success("Row updated!")
        st.subheader("Add missing item")
        new_item = st.text_input("Item name", key="new_item")
        new_price = st.number_input("Price", min_value=0.0, format="%.2f", key="new_price")
        new_cat = st.text_input("Category", value=infer_category(new_item), key="new_cat")
        if st.button("Add item"):
            if new_item.strip():
                st.session_state.edited_pairs[new_item.strip()] = new_price
                st.success(f"Added '{new_item.strip()}': {new_price:.2f}")
            else:
                st.error("Please enter a valid item name.")

        if st.button("Save all to Database"):
            current_date = date.today().isoformat()
            for item, amount in st.session_state.edited_pairs.items():
                c.execute(
                    "INSERT INTO receipts (date, item, amount, category) VALUES (?, ?, ?, ?)",
                    (current_date, item, amount, infer_category(item))
                )
            conn.commit()
            st.success("All items saved to database!")
            st.session_state.edited_pairs = {}

elif page == "Database":
    st.title("All Receipts")
    df = pd.read_sql_query("SELECT * FROM receipts", conn)
    st.dataframe(df)

elif page == "Insights":
    st.title("Insights")
    df = pd.read_sql_query("SELECT * FROM receipts", conn)
    if 'date' not in df.columns:
        st.write("No date information found in receipts.")
    else:
        df['date'] = pd.to_datetime(df['date'])
        data = df.to_dict(orient='records')
        if len(data) == 0:
            st.info("No receipts yet. Upload some!")
        else:
            st.subheader("Search Items")
            keyword = st.text_input("Search by keyword")
            if keyword:
                found = linear_search(data, keyword)[0]
                st.write("The item you were searching for:")
                st.write (f"Id: {found.get('id')}")
                st.write(f"Product purchased date: {pd.Timestamp(found.get('date')).date()}")
                st.write(f"Product Name: {found.get('item')}")
                st.write(f"Price: {found.get('amount')}")
                st.write(f"Category: {found.get('category') if 'category' in found else ''}")
            st.subheader("Sort by Amount")
            if st.button("Sort Items by Amount"):
                sorted_items = quicksort(data, 'amount')
                st.write(sorted_items)
            amounts = [r['amount'] for r in data]
            st.subheader("Aggregations")
            st.write(f"Total Spend: {sum(amounts):.2f}")
            st.write(f"Mean: {compute_mean(amounts):.2f}")
            st.write(f"Median: {compute_median(amounts):.2f}")
            st.write(f"Mode: {compute_mode(amounts):.2f}")

            st.subheader("Spending by Category")
            if 'category' in df.columns:
                cat_summary = df.groupby('category')['amount'].sum().reset_index()
                st.bar_chart(pd.Series(cat_summary.amount.values, index=cat_summary.category))

            st.subheader("Top Items")
            items = [r['item'] for r in data]
            item_counts = Counter(items)
            st.bar_chart(pd.Series(item_counts))

            st.subheader("Time-series Expenditure Trends")
            ts = df.groupby(df['date'].dt.date)['amount'].sum().reset_index()
            ts['date'] = pd.to_datetime(ts['date'])
            ts = ts.sort_values('date')
            ts.set_index('date', inplace=True)
            window = st.number_input("Moving Average window (days)", min_value=1, max_value=30, value=7)
            ts['Moving Average'] = ts['amount'].rolling(window=window, min_periods=1).mean()
            st.line_chart(ts[['amount', 'Moving Average']].rename(columns={'amount': 'Daily Spend'}))
            with st.expander("Show time-series data table"):
                st.dataframe(ts.reset_index())

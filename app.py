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

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

# --- DB SETUP (with DATE) ---
conn = sqlite3.connect('receipts.db', check_same_thread=False)
c = conn.cursor()
c.execute(
    "CREATE TABLE IF NOT EXISTS receipts (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, item TEXT, amount REAL)"
)
conn.commit()

reader = easyocr.Reader(['en'])

def extract_text(file):
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

def aggregate_word_clusters(words):
    if not words: return words
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
    s = s.replace(",", "").replace("E", "").replace("e", "").replace("O", "0")
    try:
        float(s)
        return True
    except:
        return False

def parse_items_prices(text):
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
    result = []
    for record in data:
        if keyword.lower() in record['item'].lower():
            result.append(record)
    return result

def quicksort(arr, key):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = [x for x in arr[1:] if x[key] <= pivot[key]]
    right = [x for x in arr[1:] if x[key] > pivot[key]]
    return quicksort(left, key) + [pivot] + quicksort(right, key)

def compute_mean(data):
    if len(data) == 0: return 0
    return sum(data) / len(data)

def compute_median(data):
    if len(data) == 0: return 0
    s = sorted(data)
    n = len(s)
    if n % 2 == 1:
        return s[n//2]
    else:
        return (s[n//2 - 1] + s[n//2]) / 2

def compute_mode(data):
    if len(data) == 0: return 0
    c = Counter(data)
    return c.most_common(1)[0][0]

def text_reader(file_obj):
    content = file_obj.read().decode("utf-8")
    lines = content.splitlines()
    pairs={}
    for i in range (len(lines)):
        clean_line = re.sub(r'[^\w\s]', '', lines[i])
        clean_line=clean_line.split(" ")
        product=clean_line[0]
        price=float(clean_line[-1])
        pairs[product]=pairs.get(product, 0)+price #Incase if same product purchased again then this would be usefull
    return pairs


st.sidebar.title("Receipt Processor")
page = st.sidebar.radio("Menu", ["Upload", "Database", "Insights"])

if page == "Upload":
    st.title("Upload Receipt")
    f = st.file_uploader("Upload file (.jpg, .png, .pdf, .txt)", type=['jpg', 'png', 'pdf', 'txt'])
    if f:
        if f.name.lower().endswith(".txt"):
            pairs=text_reader(f)
        else:
            text = extract_text(f)
            pairs = parse_items_prices(text)
        st.session_state.edited_pairs = pairs.copy()
        if "edited_pairs" not in st.session_state:
            st.session_state.edited_pairs = pairs.copy()
        df_display = pd.DataFrame([
            {"No.": idx+1, "Item Name": item, "Price": price}
            for idx, (item, price) in enumerate(st.session_state.edited_pairs.items())
        ])
        st.subheader("Clustered items & prices")
        st.table(df_display)
        st.markdown("### Edit a single line")
        labels = [f"{i+1} · {name}" for i, name in enumerate(st.session_state.edited_pairs)]
        label_to_item = dict(zip(labels, st.session_state.edited_pairs))
        selected = st.selectbox("Select item to edit", labels)
        orig_name = label_to_item[selected]
        orig_price = st.session_state.edited_pairs[orig_name]
        new_name  = st.text_input("Product name",  value=orig_name)
        new_price = st.number_input("Price", value=orig_price, min_value=0.0)
        if st.button("Update this row"):
            st.session_state.edited_pairs.pop(orig_name)
            st.session_state.edited_pairs[new_name] = new_price
            st.success("Row updated!")
        st.subheader("Add missing item")
        new_item = st.text_input("Item name", key="new_item")
        new_price = st.number_input("Price", min_value=0.0, format="%.2f", key="new_price")
        if st.button("Add item"):
            if new_item.strip():
                st.session_state.edited_pairs[new_item.strip()] = new_price
                st.success(f"Added “{new_item.strip()}”: ${new_price:.2f}")
            else:
                st.error("Please enter a valid item name.")
        if st.button("💾 Save all to DB"):
            current_date = date.today().isoformat()
            for item, amount in st.session_state.edited_pairs.items():
                c.execute(
                    "INSERT INTO receipts (date, item, amount) VALUES (?, ?, ?)",
                    (current_date, item, amount)
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
                st.write("The item you were wearching were of:")
                st.write (f"Id: {found.get('id')}")
                st.write(f"Product purchased date: {pd.Timestamp(found.get('date')).date()}")
                st.write(f"Product Name: {found.get('item')}")
                st.write(f"Price: {found.get('amount')}")
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

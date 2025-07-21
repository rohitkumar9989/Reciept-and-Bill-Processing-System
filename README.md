# Receipt & Bill Processing Application

A powerful data-driven mini-application for uploading, parsing, and analyzing receipts and bills—empowering users to automate data extraction, track spending, and gain financial insights from everyday transactions.

**Mail** rohitoffficial9989@gmail.com

---

## 🚀 Features

### 📂 Upload & Parsing

- **Multi-format ingestion:**  
  Upload receipts and bills as `.jpg`, `.jpeg`, `.png`, `.pdf`, or `.txt` files.
- **Automated extraction:**  
  Utilizes OCR and NLP to extract item descriptions, amounts, and dates from images, PDFs, or text files. Implementation of NLTK for proper word clustering.
- **Robust validation:**  
  Ensures only supported file types are allowed. Implements fail-safe exception handling during parsing and data processing.

### 🗃️ Data Management

- **SQLite backend:**  
  Extracted data is saved to a normalized SQLite database for reliable, persistent storage.
- **Edit before save:**  
  Review and directly edit parsed fields or add new records before committing them to the database.

### 🧠 Advanced Processing & Classic DSA

- **Custom search algorithms:**
  - **Linear search** for keyword and item discovery across uploaded records.
  - **Range and pattern-based search** for amounts and descriptions.
- **Custom sorting algorithms:**
  - **Quicksort** implementation for sorting records by numerical or categorical attributes.
- **Statistical aggregation:**
  - Calculations for **sum, mean, median, and mode** of transaction amounts.
  - **Frequency distributions**: Hash-based counting to determine most frequently purchased items.
  - **Time-series aggregation:**  
    Spend analysis by day, week, or month, with rolling averages or deltas to highlight trends.

---

## 📊 Analytics Dashboard

- **List view:**  
  Every uploaded receipt and its extracted fields (date, item, amount) is displayed in a filterable list.
- **Tabular view:**  
  All individual records with parsed fields are shown in an interactive data table, supporting sorting, searching, and editing.
- **Statistical visualizations:**  
  - **Bar/pie charts:** Categorical breakdowns reveal spending distribution and most common purchases.
  - **Time-series graphs:**  
    Line charts with moving averages showcase daily, weekly, or monthly expenditure patterns and spot trends.
- **Quick controls:**  
  Rapid search, sort, and aggregate—all using native Python data structures and explicit algorithmic implementations.

---

## 🛡️ Robustness & Extensibility

- **Validation:**  
  Strict checking for file types, schema, and input correctness.
- **Exception handling:**  
  The application catches and reports OCR, NLP, and DB errors gracefully.
- **Export-ready:**  
  Easily adaptable for exporting summaries or data as `.csv` or `.json`.
- **Multi-currency & language ready:**  
  Designed to swiftly add support for new currencies or languages.

---

## 🏗️ Technology Stack

- **Frontend/UI:** Streamlit
- **Text extraction:** EasyOCR, pdf2image, Pillow
- **PDF processing:** Requires Poppler (see limitations)
- **NLP:** NLTK
- **Backend:** SQLite
- **Visualization:** Streamlit charting and Pandas

---

## ⚡ Getting Started

### 1. Install Dependencies

`pip install streamlit easyocr pdf2image pillow pandas nltk`

*For PDF functionality, download and install Poppler from official releases and add its `bin` directory to your system Path as it is a required wheel.*

### 2. Run the App

`streamlit run app.py`

---

## 🌟 User Journey

- **Upload:** Select and upload your files. The app extracts, parses, and displays all key receipt fields. If needed, edit data or add missing items before saving.
- **Database:** Instantly browse all uploaded records in an editable table. Search and sort by any field.
- **Insights:**  
  - Search for any item or keyword.
  - Sort by price.
  - See total spend, statistical summaries, and top items.
  - Explore bar or pie charts visualizing frequent purchases and line charts displaying your spending over time (with rolling averages).

---

## 🧑‍💻 Design Highlights

- **Classic DSA approach:**  
  All searching, sorting, and data aggregation is performed using native Python logic for clarity, transparency, and educational value.
- **Modularity:**  
  Distinct modules for upload/parsing, correction, database, and visualization.
- **User empowerment:**  
  Manual correction and addition tools encourage trusted, accurate data.
- **Scalable architecture:**  
  Cleanly structured for easy extension (categories, vendor detection, multi-currency support).

---

## ⚡ Limitations & Known Issues

- **Poppler Requirement for PDFs:**  
  PDF extraction relies on the Poppler toolkit via `pdf2image`. Some users may encounter the error  
  `Exception: Unable to get page count. Is poppler installed and in PATH?`  
  when uploading a PDF.  
  **Solution:** Download and install **Poppler version 24.8**, and add its `bin` directory to your system's PATH. This resolves wheel install issues and ensures stable PDF parsing.
- **Dependence on OCR/NLP quality:**  
  Extraction accuracy may vary based on receipt clarity or layout; manual correction is included to address this.
- **Export options:**  
  Additional UI for `.csv`/`.json` export can be enabled with minimal code changes.
- **Expanded vendor/category support** and analytics possible as future enhancements.

---
## 📌 Assumptions

- **Receipts and bills are readable:** The uploaded files (images, PDFs, and text) contain clearly printed or typed text, including item descriptions, prices, and dates, making extraction via OCR and rule-based parsing feasible.
- **English language inputs:** The application is tuned to process receipts and bills primarily in English. Tokenization, OCR, and NLP routines may not be accurate with other languages unless extended.
---

*This project was made for 8Byte.ai*

---

**Simplify your spending: upload, analyze, and gain deep insight into your receipts and bills—all in one app.**

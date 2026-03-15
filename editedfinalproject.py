
# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import matplotlib
matplotlib.use("TkAgg")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 2. LOAD DATASET
df = pd.read_csv(r"D:/project1/library_custom_dataset_.csv")
df = df.fillna("")

# DATE & TIME PREPROCESSING
date_columns = ['BookTakenDate', 'BookReturnDate']
time_columns = ['InTime', 'OutTime']

for col in date_columns:
    df[col] = pd.to_datetime(df[col], format="%d-%m-%Y", errors="coerce")

def parse_time(t):
    formats = ["%H:%M:%S", "%H:%M", "%I:%M %p"]
    for f in formats:
        try:
            return pd.to_datetime(t, format=f).time()
        except:
            continue
    return None

for col in time_columns:
    df[col] = df[col].astype(str).apply(parse_time)

print("Preprocessing Completed")

# DATA VISUALIZATIONS 

sns.set(style="whitegrid")

# Books Taken Over Time
plt.figure(figsize=(10,5))
df.groupby(df['BookTakenDate'].dt.date)['BookID'].count().plot()
plt.title("Books Taken Over Time")
plt.xlabel("Date")
plt.ylabel("Books Count")
plt.tight_layout()  # Prevent overlapping
plt.show()

# Top Book Categories
plt.figure(figsize=(10,6))
cat_counts = df['BookCategory'].value_counts().head(5)

sns.barplot(
    x=cat_counts.values,
    y=cat_counts.index,
    hue=cat_counts.index,   
    palette="coolwarm",
    legend=False
)
plt.title("Top 5 Most Borrowed Book Categories")
plt.xlabel("Books Borrowed")
plt.ylabel("Book Category")
plt.tight_layout()
plt.show()


# User Type Distribution
plt.figure(figsize=(6,6))
df['UserType'].value_counts().plot.pie(autopct="%1.1f%%")
plt.title("User Type Distribution")
plt.ylabel("")
plt.show()

#4. Monthly Borrow Trend

borrow_trend = df.groupby(df['BookTakenDate'].dt.to_period("M")).size()
plt.figure(figsize=(10,6))
borrow_trend.plot(kind="line", marker="o", linestyle='-', color='skyblue')
plt.title("Monthly Borrowing Trend")
plt.xlabel("Month")
plt.ylabel("Books Borrowed")
plt.grid(True)
plt.tight_layout()
plt.show()


# 5. Countplot – Borrowing by Semester (0.14-safe)

sns.countplot(
    data=df,
    x="Semester",
    hue="Semester",      # FIX
    palette="cubehelix",
    legend=False
)
plt.title("Borrowing by Semester")
plt.show()

# 6. BookCategory by UserType Countplot

sns.countplot(
    data=df,
    x="BookCategory",
    hue="UserType",
    palette="Set2",
    order=df['BookCategory'].value_counts().index
)
plt.title("Book Categories Borrowed by User Type")
plt.xticks(rotation=45)
plt.show()

# 7. Bubble Chart – Availability by Semester & Department
plt.figure(figsize=(12,6))
sns.scatterplot(
    data=df,
    x="Semester",
    y="Department",
    size="BalanceBookAvailability",
    hue="BookCategory",
    alpha=0.6,
    sizes=(50, 500),  # Adjust min and max bubble size
    edgecolor="black"
)
plt.title("Bubble Chart: Availability by Semester & Department")
plt.tight_layout()
plt.show()
# 8. Donut Chart – Top Authors

author_counts = df['BookAuthor'].value_counts().head(8)
plt.figure(figsize=(7,7))
plt.pie(
    author_counts,
    labels=author_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=sns.color_palette("pastel"),
    wedgeprops=dict(width=0.4)
)
plt.title("Top Authors Distribution (Donut Chart)")
plt.show()

# 9. ECDF Plot – Borrowing Date Distribution

plt.figure(figsize=(10,6))
sns.ecdfplot(df['BookTakenDate'].dropna())
plt.title("ECDF Plot: Book Borrowing Date Distribution")
plt.xlabel("Borrow Date")
plt.ylabel("ECDF")
plt.show()

# TEXT FEATURE ENGINEERING

df = df.fillna("")
df["Combined"] = df["BookName"] + " " + df["BookAuthor"] + " " + df["BookCategory"]

vectorizer = TfidfVectorizer(stop_words="english")
X_tfidf = vectorizer.fit_transform(df["Combined"])

similarity_matrix = cosine_similarity(X_tfidf)
df["SimilarityScore"] = similarity_matrix.sum(axis=1)

# CREATE TARGET LABELS
q1 = df["SimilarityScore"].quantile(0.30)
q2 = df["SimilarityScore"].quantile(0.70)

def create_label(score):
    if score >= q2:
        return "SHOULD_BUY"
    elif score >= q1:
        return "MAY_BUY"
    else:
        return "NOT_BUY"

df["PurchaseLabel"] = df["SimilarityScore"].apply(create_label)
print("\nPurchase Label Distribution:")
print(df["PurchaseLabel"].value_counts())

# TRAIN–TEST SPLIT
X = X_tfidf
y = df["PurchaseLabel"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TRAIN MODEL
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("\nModel Trained Successfully ")

y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 3. TEXT PROCESSING

df["Combined"] = (
    df["BookName"] + " " +
    df["BookAuthor"] + " " +
    df["BookCategory"]
)

vectorizer = TfidfVectorizer(stop_words="english")
X_tfidf = vectorizer.fit_transform(df["Combined"])

# 4. SIMILARITY + LABEL

similarity = cosine_similarity(X_tfidf)
df["SimilarityScore"] = similarity.sum(axis=1)

q1 = df["SimilarityScore"].quantile(0.30)
q2 = df["SimilarityScore"].quantile(0.70)

def label(score):
    if score >= q2:
        return "SHOULD BUY"
    elif score >= q1:
        return "MAY BUY"
    else:
        return "DO NOT BUY"

df["PurchaseLabel"] = df["SimilarityScore"].apply(label)

# 5. TRAIN MODEL

X = X_tfidf
y = df["PurchaseLabel"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = int(accuracy_score(y_test, model.predict(X_test)) * 100)


# TKINTER UI

root = tk.Tk()
root.title("Library Book Purchase Recommendation System")
root.geometry("1250x700")
root.configure(bg="#ECEFF4")

SIDEBAR = "#1F2933"
PRIMARY = "#2563EB"
CARD = "#FFFFFF"

TITLE_FONT = ("Segoe UI", 18, "bold")
HEADER_FONT = ("Segoe UI", 14, "bold")
LABEL_FONT = ("Segoe UI", 11)

# HEADER

header = tk.Frame(root, bg=PRIMARY, height=60)
header.pack(fill="x")

tk.Label(
    header,
    text="Library Book Purchase Recommendation System",
    bg=PRIMARY,
    fg="white",
    font=TITLE_FONT
).pack(pady=12)

# BODY

body = tk.Frame(root, bg="#ECEFF4")
body.pack(fill="both", expand=True)

sidebar = tk.Frame(body, bg=SIDEBAR, width=320)
sidebar.pack(side="left", fill="y")
sidebar.pack_propagate(False)

content = tk.Frame(body, bg="#ECEFF4")
content.pack(side="right", fill="both", expand=True, padx=20, pady=20)

# SIDEBAR INPUTS

tk.Label(
    sidebar,
    text="BOOK INPUT",
    bg=SIDEBAR,
    fg="white",
    font=HEADER_FONT
).pack(pady=15)

def sidebar_field(label):
    tk.Label(
        sidebar,
        text=label,
        bg=SIDEBAR,
        fg="#CBD5E1",
        font=LABEL_FONT
    ).pack(anchor="w", padx=25, pady=(10, 0))

    entry = tk.Entry(sidebar, font=LABEL_FONT, width=25)
    entry.pack(padx=25, pady=5)
    return entry

entry_book = sidebar_field("Book Name")
entry_author = sidebar_field("Author Name")
entry_category = sidebar_field("Category")

result_box = tk.Frame(sidebar, bg="#111827")
result_box.pack(padx=20, pady=25, fill="x")

result_label = tk.Label(
    result_box,
    text="Waiting for input...",
    bg="#111827",
    fg="#E5E7EB",
    justify="left"
)
result_label.pack(padx=10, pady=10)

# =========================
# GRAPH AREA
# =========================
graph_card = tk.Frame(content, bg=CARD)
graph_card.pack(fill="both", expand=True)

tk.Label(
    graph_card,
    text="Library Borrowing Analysis",
    bg=CARD,
    font=HEADER_FONT
).pack(pady=10)

canvas_frame = tk.Frame(graph_card, bg=CARD)
canvas_frame.pack(fill="both", expand=True)

def clear_graph():
    for widget in canvas_frame.winfo_children():
        widget.destroy()

# =========================
# REAL MONTHLY BORROW GRAPH
# =========================
def show_monthly_graph(book_name):
    data = df[
        (df["BookName"].str.lower() == book_name.lower()) &
        (df["BookTakenDate"].notna())
    ]

    if data.empty:
        tk.Label(
            canvas_frame,
            text="No borrowing history available",
            bg=CARD,
            fg="gray"
        ).pack(pady=40)
        return

    data = data.copy()
    data["Month"] = data["BookTakenDate"].dt.month

    counts = (
        data.groupby("Month")
        .size()
        .reindex(range(1, 13), fill_value=0)
    )

    months = [calendar.month_abbr[m] for m in range(1, 13)]

    avg = np.mean(counts.values)   

    fig, ax = plt.subplots(figsize=(10.5, 6), dpi=100)
    ax.plot(months, counts, marker="o", linewidth=3)
    ax.fill_between(months, counts, alpha=0.25)

    ax.axhline(
        avg,
        linestyle="--",
        linewidth=2,
        label=f"Average ({avg:.1f})"
    )

    ax.set_title(f"Monthly Borrow Trend – {book_name} in 2025")
    ax.set_xlabel("Month")
    ax.set_ylabel("Borrow Count")
    ax.grid(alpha=0.3)
    ax.legend()

    ax.tick_params(axis="x", rotation=30)
    fig.subplots_adjust(bottom=0.25)

    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    plt.close(fig)


# GROUPED BAR CHART
def show_grouped_bar_chart():
    data = df[
        (df["Semester"].notna()) &
        (df["BookCategory"].notna())
    ]

    grouped = (
        data.groupby(["Semester", "BookCategory"])
        .size()
        .unstack(fill_value=0)
    )

    semesters = grouped.index.astype(str)
    categories = grouped.columns

    x = np.arange(len(semesters))
    bar_width = 0.8 / len(categories)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=100)

    for i, category in enumerate(categories):
        ax.bar(
            x + i * bar_width,
            grouped[category],
            width=bar_width,
            label=category
        )

    ax.set_xlabel("Semester")
    ax.set_ylabel("Books Borrowed")
    ax.set_title("Semester-wise Borrow Demand  in 2025 by Book Category")

    ax.set_xticks(x + bar_width * (len(categories) - 1) / 2)
    ax.set_xticklabels(semesters)

    ax.legend(title="Category")
    ax.grid(axis="y", alpha=0.3)

    fig.subplots_adjust(bottom=0.25)

    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    plt.close(fig)

# PREDICTION
COLOR_MAP = {
    "SHOULD BUY": "#16A34A",
    "MAY BUY": "#F59E0B",
    "DO NOT BUY": "#DC2626"
}

def predict():
    book = entry_book.get().strip()
    author = entry_author.get().strip()
    category = entry_category.get().strip()

    if not book or not author or not category:
        messagebox.showerror("Error", "Please fill all fields")
        return

    vec = vectorizer.transform([f"{book} {author} {category}"])
    decision = model.predict(vec)[0]
    confidence = round(max(model.predict_proba(vec)[0]) * 100, 2)

    result_label.config(
        text=(
            f"📘 {book}\n"
            f"✍ {author}\n"
            f"📚 {category}\n\n"
            f"Decision : {decision}\n"
            f"Accuracy : {accuracy}%\n"
            f"Confidence : {confidence}%"
        ),
        fg=COLOR_MAP[decision]
    )

    clear_graph()
    show_monthly_graph(book)

# BUTTONS

tk.Button(
    sidebar,
    text="Get Recommendation",
    bg=PRIMARY,
    fg="white",
    font=("Segoe UI", 11, "bold"),
    relief="flat",
    command=predict
).pack(pady=8)
tk.Button(
    sidebar,
    text="Semester-wise Demand",
    bg="#0EA5E9",
    fg="white",
    font=("Segoe UI", 10, "bold"),
    relief="flat",
    command=lambda: (clear_graph(), show_grouped_bar_chart())
).pack(pady=6)

# FOOTER
tk.Label(
    root,
    text="© 2026 | Library Recommendation System",
    bg="#ECEFF4",
    fg="gray"
).pack(pady=5)

# RUN
root.mainloop()

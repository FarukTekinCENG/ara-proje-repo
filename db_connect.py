from dotenv import load_dotenv
import os
import sqlite3

# .env dosyasını yükle
load_dotenv()

db_path = os.getenv("DB_PATH")
print("DB_PATH:", db_path)  # test için

if db_path is None:
    raise ValueError("DB_PATH environment variable is not set!")

conn = sqlite3.connect(db_path)

# 2️⃣ Cursor oluştur (SQL sorguları için)
cur = conn.cursor()

# 3️⃣ Tablo oluşturma örneği
cur.execute("""
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    age INTEGER,
    department TEXT
)
""")

# 4️⃣ Veri ekleme örneği
cur.execute("INSERT INTO employees (name, age, department) VALUES (?, ?, ?)", 
            ("Alice", 30, "IT"))
cur.execute("INSERT INTO employees (name, age, department) VALUES (?, ?, ?)", 
            ("Bob", 25, "HR"))

# 5️⃣ Değişiklikleri kaydet
conn.commit()

# 6️⃣ Veri sorgulama
cur.execute("SELECT * FROM employees")
rows = cur.fetchall()
for row in rows:
    print(row)

# 7️⃣ Bağlantıyı kapat
conn.close()

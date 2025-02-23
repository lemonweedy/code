import pandas as pd
import matplotlib.pyplot as plt


data = {
    "Item": ["Sepatu Bola", "Jersey", "Bola Basket", "Raket", "Sarung Tangan"],
    "Sales": [120, 90, 75, 50, 30]
}

df = pd.DataFrame(data)


print("Pilih jenis grafik: line, pie, bar, barh, hist")
chart_type = input("Masukkan pilihan grafik: ").strip().lower()

plt.figure(figsize=(8, 5))

if chart_type == "line":
    plt.plot(df["Item"], df["Sales"], marker='o', linestyle='-')
    plt.xlabel("Item")
    plt.ylabel("Sales")
    plt.title("Grafik Penjualan (Line)")

elif chart_type == "pie":
    plt.pie(df["Sales"], labels=df["Item"], autopct='%1.1f%%', startangle=90)
    plt.title("Grafik Penjualan (Pie)")

elif chart_type == "bar":
    plt.bar(df["Item"], df["Sales"], color='blue')
    plt.xlabel("Item")
    plt.ylabel("Sales")
    plt.title("Grafik Penjualan (Bar)")

elif chart_type == "barh":
    plt.barh(df["Item"], df["Sales"], color='green')
    plt.xlabel("Sales")
    plt.ylabel("Item")
    plt.title("Grafik Penjualan (Horizontal Bar)")

elif chart_type == "hist":
    plt.hist(df["Sales"], bins=5, color='red', alpha=0.7)
    plt.xlabel("Sales")
    plt.ylabel("Frequency")
    plt.title("Grafik Penjualan (Histogram)")

else:
    print("Pilihan tidak valid!")

plt.show()

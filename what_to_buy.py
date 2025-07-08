import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime, timedelta

# Ustawienia strategii i symulacji
COMMISSION_RATE = 0.0002  # Prowizja 0.02%
initial_cash = 10000.0    # Kapitał startowy
first_investment_date = pd.to_datetime("2025-04-02")  # Pierwsza możliwa inwestycja
FIXED_INVESTMENT = 2500.0  # Stała kwota inwestycji na kupno

# Nazwa pliku logu, w którym będą zapisywane wszystkie transakcje
log_filename = "tranzakcje_co_kupic.txt"

# Jeśli plik logu już istnieje, usuwamy go, aby rozpocząć nową historię
if os.path.exists(log_filename):
    os.remove(log_filename)

# Wczytanie danych ze wszystkich plików w folderze "akcje"
script_dir = os.path.dirname(os.path.abspath(__file__))  # Pobiera ścieżkę do katalogu skryptu
ticker_files = glob.glob(os.path.join(script_dir, "akcje_pl", "*.txt"))
events = []         # Lista zdarzeń transakcyjnych: {date, ticker, signal, price}
ticker_data = {}    # Słownik przechowujący DataFrame dla każdego tickera

for file in ticker_files:
    with open(file, 'r') as f:
        first_line = f.readline().strip()

    if "<" in first_line and ">" in first_line:
        df = pd.read_csv(file, header=0)
        df.columns = [col.strip("<>").strip().upper() for col in df.columns]
    else:
        df = pd.read_csv(file, header=None,
                         names=["TICKER", "PER", "DATE", "TIME", "OPEN", "HIGH", "LOW", "CLOSE", "VOL", "OPENINT"])

    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d", errors='coerce')
    df = df.dropna(subset=["DATE"])
    df.sort_values("DATE", inplace=True)

    # Obliczenie średnich SMA20 i SMA110 tylko dla danych od 2018 roku
    df_pre_2018 = df[df["DATE"] < "2018-01-01"].copy()
    df_from_2018 = df[df["DATE"] >= "2018-01-01"].copy()
    df_from_2018["SMA20"] = df_from_2018["CLOSE"].rolling(window=20).mean()
    df_from_2018["SMA110"] = df_from_2018["CLOSE"].rolling(window=90).mean()

    # Połączenie danych przed i po 2018 roku
    df = pd.concat([df_pre_2018, df_from_2018]).sort_values("DATE")

    # Potwierdzenie wolumenem (próg 1.25x) od 2018 roku
    df_pre_2018_vol = df[df["DATE"] < "2018-01-01"].copy()
    df_from_2018_vol = df[df["DATE"] >= "2018-01-01"].copy()
    df_from_2018_vol["VOL_SMA20"] = df_from_2018_vol["VOL"].rolling(window=20).mean()  # SMA dla wolumenu
    df_from_2018_vol["Volume_Confirm"] = df_from_2018_vol["VOL"] > (df_from_2018_vol["VOL_SMA20"] * 1.25)
    df = pd.concat([df_pre_2018_vol, df_from_2018_vol]).sort_values("DATE")

    # Różnica między średnimi w procentach (0.5%) od 2018 roku
    df.loc[df["DATE"] >= "2018-01-01", "MA_Diff_Percent"] = abs(df["SMA20"] - df["SMA110"]) / df["SMA110"] * 100

    # Obliczenie RSI
    df["Change"] = df["CLOSE"].diff()
    df["Gain"] = df["Change"].apply(lambda x: x if x > 0 else 0)
    df["Loss"] = df["Change"].apply(lambda x: -x if x < 0 else 0)
    df["Avg_Gain"] = df["Gain"].rolling(window=14).mean()
    df["Avg_Loss"] = df["Loss"].rolling(window=14).mean()
    df["RS"] = df["Avg_Gain"] / df["Avg_Loss"]
    df["RSI"] = 100 - (100 / (1 + df["RS"]))

    df["Signal"] = None
    df.reset_index(drop=True, inplace=True)
    for i in range(1, len(df)):
        if (pd.notna(df.loc[i-1, "SMA20"]) and pd.notna(df.loc[i-1, "SMA110"]) and 
            pd.notna(df.loc[i, "SMA20"]) and pd.notna(df.loc[i, "SMA110"])):
            if (df.loc[i-1, "SMA20"] < df.loc[i-1, "SMA110"] and 
                df.loc[i, "SMA20"] >= df.loc[i, "SMA110"] and 
                df.loc[i, "Volume_Confirm"] and 
                df.loc[i, "MA_Diff_Percent"] > 0.5 and 
                df.loc[i, "RSI"] < 80):
                df.loc[i, "Signal"] = "buy"
            elif (df.loc[i-1, "SMA20"] > df.loc[i-1, "SMA110"] and 
                  df.loc[i, "SMA20"] <= df.loc[i, "SMA110"] and 
                  df.loc[i, "Volume_Confirm"] and 
                  df.loc[i, "MA_Diff_Percent"] > 0.5 and 
                  df.loc[i, "RSI"] > 20):
                df.loc[i, "Signal"] = "sell"

    if not df.empty:
        ticker = df.iloc[0]["TICKER"]
        ticker_data[ticker] = df.copy()

        ticker_events = df.dropna(subset=["Signal"])[["DATE", "Signal", "CLOSE"]].copy()
        ticker_events["Ticker"] = ticker
        for idx, row in ticker_events.iterrows():
            event = {
                "date": row["DATE"],
                "ticker": ticker,
                "signal": row["Signal"],
                "price": row["CLOSE"]
            }
            events.append(event)

# Symulacja portfela
cash = initial_cash
positions = {}  # Słownik: ticker -> liczba akcji
portfolio_values = []  # Lista wartości portfela w czasie
portfolio_dates = []

events.sort(key=lambda x: x["date"])  # Sortowanie zdarzeń chronologicznie

for event in events:
    date = event["date"]
    ticker = event["ticker"]
    price = event["price"]
    signal = event["signal"]

    # Obliczanie aktualnej wartości portfela
    portfolio_value = cash
    for pos_ticker, shares in positions.items():
        current_price = ticker_data[pos_ticker][ticker_data[pos_ticker]["DATE"] <= date]["CLOSE"].iloc[-1]
        portfolio_value += shares * current_price
    portfolio_values.append(portfolio_value)
    portfolio_dates.append(date)

    # Wykonywanie transakcji tylko od 21.03.2025
    if date >= first_investment_date:
        if signal == "buy" and ticker not in positions and cash >= FIXED_INVESTMENT:
            # Kupno: używamy stałej kwoty 2500 USD
            investment = FIXED_INVESTMENT
            commission = investment * COMMISSION_RATE
            shares = (investment - commission) / price
            positions[ticker] = shares
            cash -= (investment + commission)
            with open(log_filename, "a") as log:
                log.write(f"{date}: BUY {ticker}, Shares: {shares:.2f}, Price: {price:.2f}, Cash left: {cash:.2f}\n")

        elif signal == "sell" and ticker in positions:
            # Sprzedaż: sprzedajemy wszystkie posiadane akcje
            shares = positions[ticker]
            proceeds = shares * price
            commission = proceeds * COMMISSION_RATE
            cash += proceeds - commission
            del positions[ticker]
            with open(log_filename, "a") as log:
                log.write(f"{date}: SELL {ticker}, Shares: {shares:.2f}, Price: {price:.2f}, Cash left: {cash:.2f}\n")

# Obliczenie końcowej wartości portfela
final_value = cash
for ticker, shares in positions.items():
    last_price = ticker_data[ticker]["CLOSE"].iloc[-1]
    final_value += shares * last_price

# Obliczenie średniej rocznej stopy zwrotu od 21.03.2025
start_date = first_investment_date
end_date = max(df["DATE"].max() for df in ticker_data.values())
years = (end_date - start_date).days / 365.25
cagr = (final_value / initial_cash) ** (1 / years) - 1 if years > 0 else 0

# Wyświetlanie wyników
print(f"Liczba wykrytych zdarzeń: {len(events)}")
print(f"Wartość początkowa portfela: ${initial_cash:.2f}")
print(f"Wartość końcowa portfela: ${final_value:.2f}")
print(f"Okres inwestycji (od {start_date.date()}): {years:.2f} lat")
print(f"Średnia roczna stopa zwrotu (CAGR): {cagr:.2%}")

# Dopisanie CAGR do pliku tranzakcje.txt
with open(log_filename, "a") as log:
    log.write(f"\nKońcowa wartość portfela: ${final_value:.2f}\n")
    log.write(f"Średnia roczna stopa zwrotu (CAGR): {cagr:.2%}\n")

# Sygnały dla ostatnich 5 dni
if ticker_data:
    latest_date = max(df["DATE"].max() for df in ticker_data.values())
    last_5_days = pd.date_range(end=latest_date, periods=5, freq='B')

    for date in last_5_days:
        buy_signals = [ticker for ticker, df in ticker_data.items() if df.loc[df["DATE"] == date, "Signal"].eq("buy").any()]
        sell_signals = [ticker for ticker, df in ticker_data.items() if df.loc[df["DATE"] == date, "Signal"].eq("sell").any()]

        print(f"\nData: {date.date()}")
        print(f"Sygnały KUPNA: {', '.join(buy_signals) if buy_signals else 'Brak'}")
        print(f"Sygnały SPRZEDAŻY: {', '.join(sell_signals) if sell_signals else 'Brak'}")
else:
    print("Brak dostępnych danych do analizy.")

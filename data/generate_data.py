"""
Synthetic shipment data generator for the delay prediction model.

Generates shipment-level records with route, carrier, weather, congestion,
and operational features, plus a binary `is_delayed` target whose
probability is driven by a weighted combination of those features
(passed through a logistic squashing function for realistic separability).
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

N = 8000

# (origin, destination, base_transit_days, route_risk 0-1)
ROUTES = [
    ("Shanghai", "Los Angeles", 14, 0.35),
    ("Rotterdam", "New York", 10, 0.25),
    ("Singapore", "Hamburg", 18, 0.45),
    ("Dubai", "Mumbai", 5, 0.15),
    ("Busan", "Seattle", 12, 0.30),
    ("Hong Kong", "London", 20, 0.50),
    ("Guangzhou", "Sydney", 15, 0.28),
    ("Antwerp", "Chicago", 13, 0.20),
]

# carrier -> baseline risk (higher = less reliable)
CARRIERS = {
    "MaerskLine": 0.10,
    "MSC": 0.20,
    "COSCO": 0.35,
    "EverGreen": 0.28,
    "HapagLloyd": 0.15,
}

# weather -> severity / risk contribution
WEATHER_CONDITIONS = {
    "Clear": 0.02,
    "Cloudy": 0.10,
    "Rainy": 0.30,
    "Storm": 0.65,
    "Hurricane": 0.95,
}


def generate_shipment_data(n=N):
    records = []
    start_date = datetime(2022, 1, 1)

    for i in range(n):
        route = random.choice(ROUTES)
        origin, destination, base_transit, route_risk = route

        carrier = random.choice(list(CARRIERS.keys()))
        carrier_risk = CARRIERS[carrier]

        weather = random.choice(list(WEATHER_CONDITIONS.keys()))
        weather_risk = WEATHER_CONDITIONS[weather]

        port_congestion = np.random.beta(2, 5)  # 0-1 score, mean ~0.29
        customs_complexity = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
        cargo_weight_tons = np.random.lognormal(mean=3, sigma=1)
        num_stops = np.random.choice([0, 1, 2, 3], p=[0.5, 0.3, 0.15, 0.05])
        is_peak_season = random.random() < 0.3
        carrier_on_time_rate = float(
            np.clip(0.92 - carrier_risk + np.random.normal(0, 0.03), 0.5, 0.99)
        )
        days_since_maintenance = np.random.randint(0, 180)
        fuel_cost_index = np.random.uniform(0.7, 1.3)

        ship_date = start_date + timedelta(days=random.randint(0, 730))

        # ── Latent risk score (linear combination of standardized-ish drivers) ──
        risk_score = (
            2.2 * route_risk
            + 2.0 * weather_risk
            + 1.8 * port_congestion
            + 0.9 * carrier_risk
            + 0.5 * (customs_complexity / 3)
            + 0.6 * (num_stops / 3)
            + 0.7 * (1 - carrier_on_time_rate)
            + 0.4 * is_peak_season
            + 0.3 * (days_since_maintenance / 180)
        )

        # Center and squash through a logistic function for a realistic
        # probability distribution (avoids "almost everything ~0.3-0.5")
        delay_prob = 1 / (1 + np.exp(-3.2 * (risk_score - 3.05)))
        delay_prob = float(np.clip(delay_prob, 0.02, 0.98))
        is_delayed = int(np.random.random() < delay_prob)

        actual_transit = base_transit
        if is_delayed:
            delay_days = int(np.random.exponential(scale=3)) + 1
            actual_transit += delay_days

        records.append({
            "shipment_id":            f"SHP{i+1:05d}",
            "origin_port":            origin,
            "destination_port":       destination,
            "carrier":                carrier,
            "ship_date":              ship_date.strftime("%Y-%m-%d"),
            "base_transit_days":      base_transit,
            "actual_transit_days":    actual_transit,
            "port_congestion_score":  round(port_congestion, 4),
            "weather_condition":      weather,
            "customs_complexity":     customs_complexity,
            "cargo_weight_tons":      round(cargo_weight_tons, 2),
            "num_stops":              num_stops,
            "is_peak_season":         int(is_peak_season),
            "carrier_on_time_rate":   round(carrier_on_time_rate, 4),
            "days_since_maintenance": days_since_maintenance,
            "fuel_cost_index":        round(fuel_cost_index, 4),
            "is_delayed":             is_delayed,
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df = generate_shipment_data()
    df.to_csv("shipments.csv", index=False)
    print(f"Generated {len(df)} shipment records")
    print(f"Delay rate: {df['is_delayed'].mean():.1%}")
    print(df.head())

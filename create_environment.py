"""
Weather-Based PV Energy Forecasting and Simulation with Fault Injection and Market Price Integration.
"""

import openmeteo_requests
import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry
import random


def load_weather_data(start_date, end_date):
    """
    Loads 15-minute interval weather forecast data from the Open-Meteo API
    for a specified date range. Saves the result as a CSV file.

    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    """
    # Set up the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 38.74293, # IST Taguspark
        "longitude": --9.30235, # IST Taguspark
        "minutely_15": ["global_tilted_irradiance", "temperature_2m"],
        "timezone": "Europe/London",
        "start_date": start_date,
        "end_date": end_date,
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process minutely_15 data. The order of variables needs to be the same as requested.
    minutely_15 = response.Minutely15()
    minutely_15_global_tilted_irradiance = minutely_15.Variables(0).ValuesAsNumpy()
    minutely_15_temperature_2m = minutely_15.Variables(1).ValuesAsNumpy()

    minutely_15_data = {"date": pd.date_range(
        start=pd.to_datetime(minutely_15.Time(), unit="s", utc=True),
        end=pd.to_datetime(minutely_15.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=minutely_15.Interval()),
        inclusive="left"
    )}

    minutely_15_data["global_tilted_irradiance"] = minutely_15_global_tilted_irradiance
    minutely_15_data["temperature_2m"] = minutely_15_temperature_2m

    minutely_15_dataframe = pd.DataFrame(data=minutely_15_data)

    # Set 'date' as index
    minutely_15_dataframe.set_index('date', inplace=True)

    # Export to CSV
    minutely_15_dataframe.to_csv('Data/weather_data.csv', index_label='date')

    print("Data (15 min) exported to weather_conditions_5min.csv")

    return


# https://mercado.ren.pt/PT/Electr/InfoMercado/InfOp/MercOmel/paginas/precos.aspx
# €/MWh
def load_market_price(filepath):
    """
        Loads market energy price data from CSV, converts timestamps,
        filters invalid prices, and formats the columns.

        Args:
            filepath (str): Path to CSV file.

        Returns:
            pd.DataFrame: Formatted market price data with datetime index.
        """
    # Read the CSV file (adjust separator and encoding if needed)
    df = pd.read_csv(filepath, encoding="utf-8")

    # Convert 'Data' to datetime
    df["Data"] = pd.to_datetime(df["Data"], format="%Y-%m-%d")

    # Convert 'Hora' to actual time — each unit represents 15 minutes
    df["date"] = df["Data"] + pd.to_timedelta((df["Hora"] - 1) * 15, unit="min")
    df["date"] = df["date"].dt.tz_localize("UTC")
    df["date"] = df["date"].astype(str)

    # Set negative prices to zero
    df["Preco"] = df["Preco"].apply(lambda x: max(x, 0))

    # Rename column from 'Preco' to 'price'
    df = df.rename(columns={"Preco": "Price_forecast"})

    # Reorder columns to place 'datetime' first
    cols = ["date"] + [col for col in df.columns if col not in ["Data", "Hora", "date"]]
    df = df[cols]

    return df


def calculate_energy_modules(
    df_input,
    A_panel,   # Total panel area in m²
    eta_panel_base=0.20,                # Base panel efficiency (20%)
    theta_deg=10,                       # Incidence angle (degrees)
    eta_cable=0.98,                     # Cable efficiency
    eta_mppt=0.98,                      # MPPT efficiency
    delta_t_h=15/60                     # Time interval in hours (5 minutes)
):
    """
        Calculates PV energy generation based on irradiance, temperature,
        and system efficiencies.

        Args:
            df_input (pd.DataFrame): Weather input data.
            A_panel (float): Total area of PV panels (m²).
            eta_panel_base (float): Base panel efficiency.
            theta_deg (float): Incidence angle in degrees.
            eta_cable (float): Cable efficiency.
            eta_mppt (float): MPPT efficiency.
            delta_t_h (float): Time interval in hours.

        Returns:
            pd.DataFrame: Data with energy input to inverter.
        """

    df = df_input.copy()

    # Compute cosine of incidence angle
    cos_theta = np.cos(np.radians(theta_deg))

    # Calculate effective irradiance on panel surface
    df["Geff"] = df["global_tilted_irradiance"] * cos_theta

    # Adjust panel efficiency based on temperature deviation from 25°C
    temp_diff = df["temperature_2m"] - 25
    eta_temp_adjustment = np.where(
        temp_diff > 0,
        eta_panel_base * (1 - 0.005 * temp_diff),   # Decrease 0.5% per °C above 25
        eta_panel_base * (1 - (-0.004 * temp_diff)) # Increase 0.4% per °C below 25
    )

    # DC energy adjusted for temperature efficiency (Wh)
    df["EDC_adjusted"] = df["Geff"] * A_panel * eta_temp_adjustment * delta_t_h

    # Energy input to the inverter after cable and MPPT losses (Wh)
    df["E_inverter_input"] = df["EDC_adjusted"] * eta_cable * eta_mppt

    return df[['date', 'global_tilted_irradiance', 'temperature_2m', 'E_inverter_input']]


def create_faults(
    start_date,
    end_date,
    freq="15min",
    timezone="UTC"
):
    """
        Simulates random PV system faults per day by marking random time slots as faulty.

        Args:
            start_date (str): Start date.
            end_date (str): End date.
            freq (str): Frequency of intervals.
            timezone (str): Timezone for timestamps.

        Returns:
            pd.DataFrame: DataFrame with fault flags (0 = normal, 1 = fault).
        """

    all_days = pd.date_range(start=start_date, end=end_date, freq="D")
    all_faults = []

    for day in all_days:
        # Create a full-day time range for the current day
        start_datetime = pd.Timestamp(day.strftime("%Y-%m-%d") + " 00:00:00").tz_localize(timezone)
        end_datetime = pd.Timestamp(day.strftime("%Y-%m-%d") + " 23:45:00").tz_localize(timezone)
        date_range = pd.date_range(start=start_datetime, end=end_datetime, freq=freq)

        # Initialize DataFrame with all time slots marked as normal (PV_fault = 0)
        df = pd.DataFrame({"date": date_range, "PV_fault": 0})

        # Number of faults for the day (between 0 and 1)
        n_faults = random.randint(0, 1)

        # Total number of time slots in the day
        total_slots = len(df)

        for _ in range(n_faults):
            # Randomly choose a start time, ensuring the fault fits in the day's slots (max 4 slots = 60 min)
            start_slot = random.randint(0, total_slots - 4)
            duration_slots = random.randint(1, 4)  # Duration between 15 and 60 minutes

            # Mark fault in the selected range
            df.loc[start_slot:start_slot + duration_slots - 1, "PV_fault"] = 1

        # Optionally convert the 'date' column to string (as in your original code)
        df["date"] = df["date"].astype(str)

        all_faults.append(df)

    # Concatenate all daily DataFrames and return
    return pd.concat(all_faults, ignore_index=True)

import pandas as pd
import random

def create_cyberattacks(
    start_date,
    end_date,
    freq="15min",
    timezone="UTC"
    ):
    """
    Simulates random cyberattacks per day by marking random time slots as attacked.

    Args:
        start_date (str): Start date.
        end_date (str): End date.
        freq (str): Frequency of intervals.
        timezone (str): Timezone for timestamps.

    Returns:
        pd.DataFrame: DataFrame with cyberattack flags (0 = normal, 1 = cyberattack).
    """

    all_days = pd.date_range(start=start_date, end=end_date, freq="D")
    all_attacks = []

    for day in all_days:
        # Create a full-day time range for the current day
        start_datetime = pd.Timestamp(day.strftime("%Y-%m-%d") + " 00:00:00").tz_localize(timezone)
        end_datetime = pd.Timestamp(day.strftime("%Y-%m-%d") + " 23:45:00").tz_localize(timezone)
        date_range = pd.date_range(start=start_datetime, end=end_datetime, freq=freq)

        # Initialize DataFrame with all time slots marked as normal (Cyber_attack = 0)
        df = pd.DataFrame({"date": date_range, "Cyberattack_alert": 0})

        # Number of cyberattacks for the day (between 0 and 1)
        n_attacks = random.randint(0, 1)

        # Total number of time slots in the day
        total_slots = len(df)

        for _ in range(n_attacks):
            # Randomly choose a start time, ensuring the attack fits in the day's slots (max 4 slots = 60 min)
            start_slot = random.randint(0, total_slots - 4)
            duration_slots = random.randint(1, 4)  # Duration between 15 and 60 minutes

            # Mark attack in the selected range
            df.loc[start_slot:start_slot + duration_slots - 1, "Cyberattack_alert"] = 1

        # Optionally convert the 'date' column to string
        df["date"] = df["date"].astype(str)

        all_attacks.append(df)

    # Concatenate all daily DataFrames and return
    return pd.concat(all_attacks, ignore_index=True)


def buy_percentage(start_date, end_date, freq="15min", timezone="UTC"
) -> pd.DataFrame:
    """
    Generates a DataFrame with timestamps from start_date to end_date (inclusive),
    at the frequency freq, and assigns a buy_percentage according to:
      - 0.5 between 00:00–08:00 and 22:00–24:00
      - 0.75 between 08:00–09:00, 10:30–18:00, and 20:30–22:00
      - 1.0 between 09:00–10:30 and 18:00–20:30

    Returns:
      pd.DataFrame with columns ['date', 'buy_percentage'].
    """

    # Create full timestamp range for the given period
    full_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    full_range = full_range.tz_localize(timezone)

    df = pd.DataFrame({"date": full_range})

    # Convert timestamps to float-hour format (e.g., 10:30 → 10.5)
    hours = df["date"].dt.hour + df["date"].dt.minute / 60

    # Define conditions for each price tier based on hour of day
    conds = [
        (hours >= 0)   & (hours < 8)   | (hours >= 22) & (hours < 24),
        (hours >= 8)   & (hours < 9)   |
        (hours >= 10.5)& (hours < 18)  |
        (hours >= 20.5)& (hours < 22),
        (hours >= 9)   & (hours < 10.5)|
        (hours >= 18)  & (hours < 20.5)
    ]
    choices = [0.5, 0.75, 1.0]

    df["Buy_percentage"] = np.select(conds, choices, default=np.nan)
    df["date"] = df["date"].astype(str)

    return df


def create_environment(panel_area, start_date, end_date, file_weather_forecast, file_market_price, output_file):
    """
        Creates a combined dataset by merging:

          - Weather-based energy forecast (from PV model)
          - Market price data
          - Simulated fault events
          - Hourly energy buy percentage

        Steps:
          1. Load weather forecast and filter by start_date
          2. Compute forecasted inverter input energy
          3. Load market prices
          4. Simulate PV system faults
          5. Generate hourly buy percentages
          6. Merge all datasets on 'date'
          7. Export merged dataset to CSV

        Args:
            panel_area (float): Panel area.
            start_date (str): Start date.
            end_date (str): End date.
            file_weather_forecast (str): Path to weather forecast CSV.
            file_market_price (str): Path to market price CSV.
            output_file (str): Output path for final merged dataset.

        Returns:
            None
        """
    # Load past and forecast weather data
    df_forecast = pd.read_csv(file_weather_forecast)
    df_forecast = df_forecast[df_forecast['date'] >= f"{start_date} 00:00:00+00:00"]
    # Calculate energy for past and forecast weather data
    energy_forecast = calculate_energy_modules(df_forecast, panel_area)
    market_price = load_market_price(file_market_price)
    faults = create_faults(start_date, end_date)
    cyberattacks = create_cyberattacks(start_date, end_date)
    buy_perc = buy_percentage(start_date, end_date)

    # Rename forecast energy column to distinguish it
    energy_forecast = energy_forecast.rename(columns={
        "E_inverter_input": "E_inverter_input_forecast",
        "global_tilted_irradiance": "global_tilted_irradiance_forecast",
        "temperature_2m": "temperature_2m_forecast"
    })

    # 1. Merge result with market_price (inner join on date)
    merged_df = pd.merge(
        energy_forecast,
        market_price[["date", "Price_forecast"]],
        on="date",
        how="inner"
    )

    # 2. Merge result with faults (inner join on date)
    merged_df = pd.merge(
        merged_df,
        faults[["date", "PV_fault"]],
        on="date",
        how="inner"
    )

    # 3. Merge result with cyberattacks (inner join on date)
    merged_df = pd.merge(
        merged_df,
        cyberattacks[["date", "Cyberattack_alert"]],
        on="date",
        how="inner"
    )

    # 4. Merge result with buy percentage (inner join on date)
    merged_df = pd.merge(
        merged_df,
        buy_perc[["date", "Buy_percentage"]],
        on="date",
        how="inner"
    )

    # Export result to CSV
    merged_df.to_csv(output_file, index=False)

    print(f"\nEnvironment created. Data saved to {output_file}")

    return
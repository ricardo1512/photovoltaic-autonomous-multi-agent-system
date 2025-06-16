"""
Dynamic Time-Series Visualization for Energy System Simulation.
"""

import pandas as pd
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from PyQt5.QtWidgets import QApplication
import sys
import datetime

# Custom axis class for displaying datetime strings on the x-axis
class DateAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        return [
            datetime.datetime.fromtimestamp(v, tz=datetime.timezone.utc)
            .strftime("%Y-%m-%d %H:%M")
            for v in values
        ]


def create_dynamic_plots(input_file1, input_file2):
    """
    Dynamic Time-Series Visualization for Energy System Simulation.

    This script creates an interactive, real-time updating plot of multiple key metrics from
    a photovoltaic (PV) microgrid system simulation. It reads two input CSV files containing
    time-stamped data, merges them, and visualizes step and cumulative profits, lost energy,
    battery level, and other relevant parameters over time.

    Features include:
    - Custom datetime axis formatting for clear time display.
    - Logarithmic scaling for step profit values to handle a wide data range.
    - Separate plots for step values, cumulative values, battery status, and other variables.
    - Visual indicators for PV faults and cyberattack alerts as vertical dashed lines.
    - Dynamic updating of plots with smooth animation using PyQtGraph and PyQt5.

    This tool aids in analyzing system behaviour, agent performance, and fault events in PV microgrid simulations.
    """

    df1 = pd.read_csv(input_file1, parse_dates=["date"])
    df2 = pd.read_csv(input_file2, parse_dates=["date"])

    # Ensure UTC timezone
    for df in (df1, df2):
        if df["date"].dt.tz is None:
            df["date"] = df["date"].dt.tz_localize("UTC")

    # Merge both dataframes on the date column
    df = pd.merge(
        df1[["date", "E_inverter_input_forecast", "Price_forecast", "PV_fault", "Buy_percentage", "Cyberattack_alert"]],
        df2[
            [
                "date",
                "Inverter_profits_cumulative",
                "Grid_profits_cumulative",
                "Lost_energy_cumulative",
                "Grid_profits_step",
                "Inverter_profits_step",
                "Lost_energy_value_step",
                "Battery_pct",
            ]
        ],
        on="date",
        how="inner",
    ).sort_values("date")

    # Log10 conversion
    df["Inverter_profits_step"] = np.log10(df["Inverter_profits_step"].clip(lower=1e-6))
    df["Grid_profits_step"] = np.log10(df["Grid_profits_step"].clip(lower=1e-6))
    df["Lost_energy_value_step"] = np.log10(df["Lost_energy_value_step"].clip(lower=1e-6))

    # Scale cumulative values to k€
    df["Inverter_profits_cumulative"] = df["Inverter_profits_cumulative"] / 1e9
    df["Grid_profits_cumulative"] = df["Grid_profits_cumulative"] / 1e9
    df["Lost_energy_cumulative"] = df["Lost_energy_cumulative"] / 1e9

    # Scale energy forecast to kW and percentage to %
    df["E_inverter_input_forecast"] = df["E_inverter_input_forecast"] / 1e3
    df["Buy_percentage"] = df["Buy_percentage"] * 100

    profits_cols = ["Inverter_profits_step", "Grid_profits_step", "Lost_energy_value_step"]
    cumulative_cols = ["Inverter_profits_cumulative", "Grid_profits_cumulative", "Lost_energy_cumulative"]

    # Dictionary for renaming plots
    rename_dict = {
        "E_inverter_input_forecast": "Energy Forecast (kW)",
        "Price_forecast": "Price Forecast (M€)",
        "Buy_percentage": "Buy Percentage (%)",
    }

    # Collect other numeric columns, excluding Battery_pct (plotted separately)
    data_columns = df.select_dtypes(include="number").columns.tolist()
    remaining_cols = [
        c for c in data_columns if c not in profits_cols + cumulative_cols + ["PV_fault", "Cyberattack_alert", "Battery_pct"]
    ]

    # Convert timestamps to UTC seconds
    times = (df["date"].astype("int64") // 10**9).to_numpy()

    app = QApplication(sys.argv)
    glw = pg.GraphicsLayoutWidget(show=True, title="Dynamic Plotting")
    glw.resize(1000, 200 * (3 + len(remaining_cols)))  # +1 linha cumulativa +1 linha Battery_pct
    glw.setBackground("#111111")

    # --- Step profits plot (row 0) ---
    ax0 = glw.addPlot(
        row=0, col=0, title="Log10 Step PV and Grid Profits, and Lost Energy Value (€)", axisItems={"bottom": DateAxis(orientation="bottom")}
    )
    ax0.showGrid(x=True, y=True, alpha=0.3)
    # Lines: inverter, grid and lost energy (steps)
    inv_line = ax0.plot([], [], pen=pg.mkPen(pg.intColor(3, hues=12), width=2), name="PV Profits (step)")
    grd_line = ax0.plot([], [], pen=pg.mkPen(pg.intColor(2, hues=12), width=2), name="Grid Profits (step)")
    epc_line = ax0.plot([], [], pen=pg.mkPen(pg.intColor(1, hues=12), width=2), name="Energia Perdida (step)")
    # Dummy lines for faults and cyberattacks (for legend display)
    fault_line = pg.PlotDataItem([0], [0], pen=pg.mkPen("w", style=QtCore.Qt.DashLine))
    cyber_line = pg.PlotDataItem([0], [0], pen=pg.mkPen("r", style=QtCore.Qt.DashLine))
    # Manual legend to properly align text and lines
    legend = ax0.addLegend(offset=(10, 10))
    legend.setLabelTextColor("w")
    legend.addItem(inv_line, "PV")
    legend.addItem(grd_line, "Grid")
    legend.addItem(epc_line, "Lost")
    legend.addItem(fault_line, "PV Faults")
    legend.addItem(cyber_line, "Cyberattacks")
    # Lock x-axis range
    ax0.setLimits(xMin=times[0], xMax=times[-1])
    ax0.setXRange(times[0], times[-1])

    # --- Cumulative profits plot (row 1) ---
    ax1 = glw.addPlot(
        row=1, col=0, title="Cumulative PV and Grid Profits, and Lost Energy Value (k€)", axisItems={"bottom": DateAxis(orientation="bottom")}
    )
    ax1.showGrid(x=True, y=True, alpha=0.3)

    inv_cum_line = ax1.plot([], [], pen=pg.mkPen(pg.intColor(3, hues=12), width=2), name="Inverter Profits (cumulative)")
    grd_cum_line = ax1.plot([], [], pen=pg.mkPen(pg.intColor(2, hues=12), width=2), name="Grid Profits (cumulative)")
    epc_cum_line = ax1.plot([], [], pen=pg.mkPen(pg.intColor(1, hues=12), width=2), name="Lost Energy Value (cumulative)")

    legend_cum = ax1.addLegend(offset=(10, 10))
    legend_cum.setLabelTextColor("w")
    legend_cum.addItem(inv_cum_line, "PV")
    legend_cum.addItem(grd_cum_line, "Grid")
    legend_cum.addItem(epc_cum_line, "Lost")

    ax1.setLimits(xMin=times[0], xMax=times[-1])
    ax1.setXRange(times[0], times[-1])
    ax1.hideAxis("bottom")

    # --- Battery_pct plot (row 2) ---
    ax2 = glw.addPlot(
        row=2, col=0, title="Battery Level (%)", axisItems={"bottom": DateAxis(orientation="bottom")}
    )
    ax2.showGrid(x=True, y=True, alpha=0.3)
    pen_bat = pg.mkPen(pg.intColor(6, hues=12), width=2, style=QtCore.Qt.SolidLine)
    pct_line = ax2.plot([], [], pen=pen_bat, name="Battery_pct")

    # Add horizontal dashed lines at y=20 and y=80
    hline_20 = pg.InfiniteLine(pos=20, angle=0, pen=pg.mkPen('w', style=QtCore.Qt.DashLine))
    hline_80 = pg.InfiniteLine(pos=80, angle=0, pen=pg.mkPen('w', style=QtCore.Qt.DashLine))
    ax2.addItem(hline_20)
    ax2.addItem(hline_80)

    ax2.setLimits(xMin=times[0], xMax=times[-1])
    ax2.setXRange(times[0], times[-1])
    ax2.setYRange(0, 99)
    ax2.hideAxis("bottom")

    # --- Other subplots ---
    other_lines = {}
    for i, col in enumerate(remaining_cols, start=3):
        title = rename_dict.get(col, col)
        p = glw.addPlot(
            row=i,
            col=0,
            title=title,
            axisItems={
                "bottom": DateAxis(orientation="bottom")
                if i == (len(remaining_cols) + 3 - 1)
                else pg.AxisItem(orientation="bottom")
            },
        )
        p.showGrid(x=True, y=True, alpha=0.3)
        pen = pg.mkPen(pg.intColor(i + 7, hues=12), width=2)
        ln = p.plot([], [], pen=pen, name=col)
        other_lines[col] = ln
        if i < (len(remaining_cols) + 3 - 1):
            p.hideAxis("bottom")
        p.setLimits(xMin=times[0], xMax=times[-1])
        p.setXRange(times[0], times[-1])
        p.hideAxis("bottom")

        # Fix y-axis for Buy_percentage
        if col == "Buy_percentage":
            p.setYRange(0, 100)

    # Data structure for animation
    xdata = []
    ydata = {c: [] for c in profits_cols + remaining_cols + cumulative_cols + ["Battery_pct"]}

    frame = {"i": 0}

    def update():
        i = frame["i"]
        if i >= len(df):
            timer.stop()
            return
        t = times[i]
        xdata.append(t)

        # Update step profit/loss lines
        ydata["Inverter_profits_step"].append(df["Inverter_profits_step"].iloc[i])
        ydata["Grid_profits_step"].append(df["Grid_profits_step"].iloc[i])
        ydata["Lost_energy_value_step"].append(df["Lost_energy_value_step"].iloc[i])
        inv_line.setData(x=xdata, y=ydata["Inverter_profits_step"])
        grd_line.setData(x=xdata, y=ydata["Grid_profits_step"])
        epc_line.setData(x=xdata, y=ydata["Lost_energy_value_step"])

        # Update cumulative values
        ydata["Inverter_profits_cumulative"].append(df["Inverter_profits_cumulative"].iloc[i])
        ydata["Grid_profits_cumulative"].append(df["Grid_profits_cumulative"].iloc[i])
        ydata["Lost_energy_cumulative"].append(df["Lost_energy_cumulative"].iloc[i])
        inv_cum_line.setData(x=xdata, y=ydata["Inverter_profits_cumulative"])
        grd_cum_line.setData(x=xdata, y=ydata["Grid_profits_cumulative"])
        epc_cum_line.setData(x=xdata, y=ydata["Lost_energy_cumulative"])

        # Update battery level
        ydata["Battery_pct"].append(df["Battery_pct"].iloc[i])
        pct_line.setData(x=xdata, y=ydata["Battery_pct"])

        # Update other dynamic lines
        for col in remaining_cols:
            ydata[col].append(df[col].iloc[i])
            other_lines[col].setData(x=xdata, y=ydata[col])

        # Add vertical lines on fault/cyberattack events
        if df["PV_fault"].iloc[i] == 1:
            ax0.addItem(
                pg.InfiniteLine(
                    pos=t, angle=90, pen=pg.mkPen("w", style=QtCore.Qt.DashLine)
                )
            )
        if df["Cyberattack_alert"].iloc[i] == 1:
            ax0.addItem(
                pg.InfiniteLine(pos=t, angle=90, pen=pg.mkPen("r", style=QtCore.Qt.DashLine))
            )

        frame["i"] += 1

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(5)

    sys.exit(app.exec_())
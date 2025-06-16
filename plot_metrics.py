"""
Script for Comparative Visualization of Cumulative Results in Negotiation Models (Profit-Profit vs. Nash Q-learning).
"""

import pandas as pd
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
import pyqtgraph.exporters
import sys


def comparative_bar_plot(output_image_main, output_image_battery):
    """
    Generates two comparative bar plots:
    1. PV Profits, Grid Profits, and Lost Energy
    2. Battery Wear only
    and exports them as images.

    Args:
        output_image_main (str): Path where the main image will be saved.
        output_image_battery (str): Path for the battery wear image.
    """
    # File paths
    file_paths = {
        "Profit-Profit": "Data/results_neg_profit_profit.csv",
        "Nash Q-learning": "Data/results_nash.csv"
    }

    # Features for main plot
    features_main = [
        ("Inverter_profits_cumulative", "PV Revenues", "#009900"),
        ("Grid_profits_cumulative", "Grid Revenues", "#cca300"),
        ("Lost_energy_cumulative", "Lost Energy Value", "#994d00"),
    ]

    # Feature for battery wear plot
    battery_feature = ("Battery_wear_cumulative", "Battery Wear", "#cc0066")

    # Load last row from each CSV
    data_main = {}
    data_battery = {}
    for label, path in file_paths.items():
        df = pd.read_csv(path)
        last_row = df.iloc[-1]
        data_main[label] = [last_row[feat] for feat, _, _ in features_main]
        data_battery[label] = last_row[battery_feature[0]]

    # Shared setup
    x_labels = list(file_paths.keys())
    x = np.arange(len(x_labels))
    width = 0.18
    app = QApplication(sys.argv)

    # ---- MAIN PLOT ----
    win_main = pg.PlotWidget(title="Revenues and Losses for Profit-Profit Negotiation and Nash Q-learning [€]")
    win_main.setAntialiasing(True)
    win_main.setBackground('k')
    win_main.showGrid(x=True, y=True, alpha=0.3)
    win_main.setMenuEnabled(False)
    win_main.setMouseEnabled(x=False, y=False)

    max_val_main = 0
    for i, (feat, label, color) in enumerate(features_main):
        heights = [data_main[xlab][i] / 1e6 for xlab in x_labels]
        max_val_main = max(max_val_main, max(heights))
        bar = pg.BarGraphItem(x=x + (i - 1.0) * width, height=heights, width=width, brush=pg.mkBrush(color))
        win_main.addItem(bar)

    font = pg.QtGui.QFont('Arial', 4)
    for i, label in enumerate(x_labels):
        text = pg.TextItem(label, anchor=(0.5, 0.5), color='#808080')
        text.setFont(font)
        text.setPos(x[i], -0.035 * max_val_main)
        win_main.addItem(text)

    win_main.getAxis('left').setTextPen('#808080')
    win_main.getAxis('left').setTickFont(pg.QtGui.QFont('Arial', 5))
    win_main.setXRange(-0.5, len(x_labels) - 0.5)
    win_main.setYRange(0, max_val_main * 1.2)

    legend_font = pg.QtGui.QFont('Arial', 5)
    for j, (_, label, color) in enumerate(features_main):
        legend = pg.TextItem(f"■ {label}", anchor=(0, 0), color=color)
        legend.setFont(legend_font)
        legend.setPos(-0.5, max_val_main * (1.2 - 0.05 * j))
        win_main.addItem(legend)

    exporter_main = pg.exporters.ImageExporter(win_main.plotItem)
    exporter_main.parameters()['width'] = 2400
    exporter_main.parameters()['height'] = 1200
    exporter_main.export(output_image_main)

    print(f"\nComparative plot (inverter and grid profits, and lost energy value) saved to {output_image_main}")

    # ---- BATTERY WEAR PLOT ----
    win_batt = pg.PlotWidget(title="Battery Wear for Profit-Profit Negotiation and Nash Q-learning [€]")
    win_batt.setAntialiasing(True)
    win_batt.setBackground('k')
    win_batt.showGrid(x=True, y=True, alpha=0.3)
    win_batt.setMenuEnabled(False)
    win_batt.setMouseEnabled(x=False, y=False)

    heights_batt = [data_battery[xlab] / 1e3 for xlab in x_labels]
    max_val_batt = max(heights_batt)

    bar_batt = pg.BarGraphItem(x=x, height=heights_batt, width=0.3, brush=pg.mkBrush(battery_feature[2]))
    win_batt.addItem(bar_batt)

    for i, label in enumerate(x_labels):
        text = pg.TextItem(label, anchor=(0.5, 0.5), color='#808080')
        text.setFont(font)
        text.setPos(x[i], -0.035 * max_val_batt)
        win_batt.addItem(text)

    win_batt.getAxis('left').setTextPen('#808080')
    win_batt.getAxis('left').setTickFont(pg.QtGui.QFont('Arial', 5))
    win_batt.setXRange(-0.5, len(x_labels) - 0.5)
    win_batt.setYRange(0, max_val_batt * 1.2)

    label = pg.TextItem(f"■ {battery_feature[1]}", anchor=(0, 0), color=battery_feature[2])
    label.setFont(legend_font)
    label.setPos(-0.5, max_val_batt * 1.18)
    win_batt.addItem(label)

    exporter_batt = pg.exporters.ImageExporter(win_batt.plotItem)
    exporter_batt.parameters()['width'] = 2400
    exporter_batt.parameters()['height'] = 1200
    exporter_batt.export(output_image_battery)

    app.quit()

    print(f"Comparative plot (battery wear) saved to {output_image_battery}")
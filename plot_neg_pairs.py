"""
Script for Comparative Visualization of PV and Grid Profits with Lost Energy in Negotiation Models.
"""

import pandas as pd
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
import pyqtgraph.exporters
import sys


def negotiation_plot(input_file, output_image):
    """
    Generates a bar plot comparing PV profits, grid profits, and lost energy value
    across different negotiation scenarios, and exports it as an image.

    Args:
        input_file (str): Path to the CSV file containing the data.
        output_image (str): Path where the output image will be saved.
    """

    # Load and preprocess CSV data
    df = pd.read_csv(input_file)
    # Select relevant columns and create scenario labels
    df = df[['grid_mode', 'inverter_mode', 'grid_profits', 'inverter_profits', 'lost_energy_value']]
    df['scenario'] = 'PV=' + df['inverter_mode'] + '\nGRID=' + df['grid_mode']

    # Define color scheme
    grey_color = '#808080'  # Axis labels and text
    green_color = '#009900'  # PV profits
    yellow_color = '#cca300'  # Grid profits
    orange_color = '#994d00'  # Lost energy value

    # Prepare bar positions and widths
    x = np.arange(len(df))
    width = 0.2

    # Create application and plot widget
    app = QApplication(sys.argv)
    win = pg.PlotWidget(title="Results by Negotiation Scenario [€]")
    win.setAntialiasing(True)  # Smooth rendering
    win.setBackground('k')  # Black background
    win.showGrid(x=True, y=True, alpha=0.3)  # Semi-transparent grid
    win.setMenuEnabled(False)  # Disable right-click menu
    win.setMouseEnabled(x=False, y=False)  # Disable zooming/panning

    # Create bar graphs
    # PV profits bar (green, left position)
    bg1 = pg.BarGraphItem(x=x - width, height=df['inverter_profits'] / 1e6, width=width, brush=pg.mkBrush(green_color))
    # Grid profits bar (yellow, center position)
    bg2 = pg.BarGraphItem(x=x, height=df['grid_profits'] / 1e6, width=width, brush=pg.mkBrush(yellow_color))
    # Lost energy bar (orange, right position)
    bg3 = pg.BarGraphItem(x=x + width, height=df['lost_energy_value'] / 1e6, width=width,
                          brush=pg.mkBrush(orange_color))

    win.addItem(bg1)
    win.addItem(bg2)
    win.addItem(bg3)

    # Customize X-axis
    ax = win.getAxis('bottom')
    ax.setTicks([])

    # Customize Y-axis
    ax_y = win.getAxis('left')
    font_ticks = pg.QtGui.QFont('Arial', 4)
    ax_y.setTickFont(font_ticks)
    ax_y.setTextPen(grey_color)

    # Add custom scenario labels below X-axis
    font = pg.QtGui.QFont('Arial', 3)
    for i, label in enumerate(df['scenario']):
        text_item = pg.TextItem(label, anchor=(0.5, 0.5), color=grey_color)
        text_item.setFont(font)
        text_item.setPos(i, -0.05 * df[['inverter_profits', 'grid_profits', 'lost_energy_value']].values.max() / 1e6)
        win.addItem(text_item)

    # Set axis ranges
    win.setXRange(-0.5, len(df) - 0.5)
    y_min = min(0, -0.1 * df[['inverter_profits', 'grid_profits', 'lost_energy_value']].values.max() / 1e6)
    y_max = df[['inverter_profits', 'grid_profits', 'lost_energy_value']].values.max() / 1e6 * 1.2
    win.setYRange(y_min, y_max)

    # Add manual legend
    legend_font = pg.QtGui.QFont('Arial', 5)
    # PV Profits legend (green)
    label1 = pg.TextItem("■ PV Profits", anchor=(0, 0), color=green_color)
    label1.setFont(legend_font)
    label1.setPos(-0.5, y_max * 1.00)
    win.addItem(label1)
    # Grid Profits legend (yellow)
    label2 = pg.TextItem("■ Grid Profits", anchor=(0, 0), color=yellow_color)
    label2.setFont(legend_font)
    label2.setPos(-0.5, y_max * 0.95)
    win.addItem(label2)
    # Lost Energy legend (orange)
    label3 = pg.TextItem("■ Lost Energy Value", anchor=(0, 0), color=orange_color)
    label3.setFont(legend_font)
    label3.setPos(-0.5, y_max * 0.90)
    win.addItem(label3)

    exporter = pg.exporters.ImageExporter(win.plotItem)
    exporter.parameters()['width'] = 2400
    exporter.parameters()['height'] = 1200
    exporter.export(output_image)

    app.quit()

    print(f"\nNegotiation plot saved to {output_image}")
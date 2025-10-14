# Four different types of individuals to test:
from individual import QChromosomeRepairPermutationEncoding, QChromosomePositionEncoding, QChromosomeHashPermutationEncoding, QChromosomeHashMultisetEncoding, QChromosomeHashMultisetImprovedEncoding

# Importing libraries
import numpy as np
import pandas as pd
import pdb
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import seaborn as sns
from typing import List
import math
import decimal
import json
from sympy.combinatorics.graycode import bin_to_gray, gray_to_bin
import matplotlib.animation as animation

def plot_rotation_gate(fig, ax, a, b, a_previous, b_previous):
    theta_initial = np.arccos(a_previous)
    theta_new = np.arccos(a)
    x_arc = np.cos(np.linspace(np.minimum(theta_initial, theta_new), np.maximum(theta_initial, theta_new), 20))*0.25
    y_arc = np.sin(np.linspace(np.minimum(theta_initial, theta_new), np.maximum(theta_initial, theta_new), 20))*0.25

    print("0: " + str(abs(a)**2))
    print("1: " + str(abs(b)**2))
    print(f"sum = {abs(a)**2 + abs(b)**2}")

    xaxis = plt.Line2D((-1.3, 1.3), (0, 0), c="black")
    yaxis = plt.Line2D((0, 0), (-1.3, 1.3), c="black")

    circle_1 = plt.Circle((0, 0), 1, fill=False)
    arrow_1 = plt.Arrow(0, 0, a, b, width=0.1, color="green")
    arrow_2 = plt.Arrow(0, 0, a_previous, b_previous, width=0.1,
                        edgecolor=(1, 122/255, 112/255, 1), 
                        linestyle="--",
                        facecolor=(247/255, 183/255, 178/255, 0.2)
                    )

    ax.set_ylim(-1.3, 1.3)
    ax.set_xlim(-1.3, 1.3)
    ax.add_patch(circle_1)
    ax.add_line(xaxis)
    ax.add_line(yaxis)
    ax.add_patch(arrow_1)
    
    #if a != a_previous or b != b_previous:
    #    ax.add_patch(arrow_2)

    ax.text(0.05, 1.07, r"$|1\rangle$")
    ax.text(0.05, -1.1, r"$|1\rangle$")
    ax.text(-1.2, 0.05, r"$|0\rangle$")
    ax.text(1.05, 0.05, r"$|0\rangle$")
    ax.text(0.3, 0.3, r"$ab > 0$")
    ax.text(-0.5, -0.3, r"$ab > 0$")
    ax.text(-0.5, 0.3, r"$ab < 0$")
    ax.text(0.3, -0.3, r"$ab < 0$")

    #ax.text(-1.1, 1.1, f"{abs(a)**2:.2f} " + r"$|0\rangle$" + f" + {abs(b)**2:.2f} " + r"$|1\rangle$")
    ax.text(-1.1, 1.1, r"p_0: " + f"{abs(a)**2:.2f}")
    ax.text(-1.1, 0.9, r"p_1: " + f"{abs(b)**2:.2f}")
    return ax.plot(x_arc, y_arc, color="r", linewidth=0.9)
    


a = a_previous = 1/np.sqrt(2)
b = b_previous = 1/np.sqrt(2)

artists = []
for i in range(20):
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_rotation_gate(fig, ax, a, b, a_previous, b_previous)
    plt.savefig(f"rotation_gif_imgs/img_{i}.png")

    angle = 0.02*np.pi
    a_temp = a*np.cos(angle) - b*np.sin(angle)
    b_temp = a*np.sin(angle) + b*np.cos(angle)
    a_previous = a
    b_previous = b
    a = a_temp
    b = b_temp


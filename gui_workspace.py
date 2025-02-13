import tkinter as tk
import time
from matplotlib.figure import Figure 
from matplotlib.pyplot import subplots
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
from functools import *

import numpy as np

from matplotlib import style

import threading

class TestShared:
    shared_val = 0

style.use('ggplot')

# the figure that will contain the plot 
fig, ax = subplots(1, 1, figsize = (5, 5), 
                dpi = 100) 

# adding the subplot 
#plot1 = fig.add_subplot(111) 

def test_func(i, shared_obj: TestShared):
    for k in range(i):
        time.sleep(2)
        shared_obj.shared_val += k
        print(shared_obj.shared_val)

def animate(i, shared_obj):
    # list of squares 
    x = np.linspace(0, 10, 1000)
    #bounds = np.random.randint(-5, 10, 2)
    y = np.ones(1000)*shared_obj.shared_val #bounds[0], bounds[1], 1000)

    # plotting the graph 
    ax.clear()
    ax.plot(x, y) 

# https://stackoverflow.com/questions/69960432/tkinter-mainloop-not-quitting-after-closing-window
def _quit():
    window.quit()
    window.destroy() 


shared_obj = TestShared()

# the main Tkinter window 
window = tk.Tk() 
window.protocol("WM_DELETE_WINDOW", _quit)
# setting the title  
window.title('Plotting in Tkinter') 
  
# dimensions of the main window 
window.geometry("700x700") 

# creating the Tkinter canvas 
# containing the Matplotlib figure 
canvas = FigureCanvasTkAgg(fig, 
                            master = window)   
canvas.draw() 

# placing the canvas on the Tkinter window 
canvas.get_tk_widget().pack() 

ani = animation.FuncAnimation(fig, partial(animate, shared_obj=shared_obj), interval=1000, cache_frame_data=False)

#for i in range(1000):
#    time.sleep(1)
#    print(i)
# run the gui 
test_thread = threading.Thread(target=test_func, args=(5, shared_obj))  
test_thread.start()
window.mainloop()

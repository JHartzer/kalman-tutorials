''' Present an interactive function explorer with slider widgets.

Scrub the sliders to change the properties of the ``sin`` curve, or
type into the title text box to update the title of the plot.

Use the ``bokeh serve`` command to run the example by executing:

    bokeh serve sliders.py

at your command prompt. Then navigate to the URL

    http://localhost:5006/sliders

in your browser.

'''
import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider
from bokeh.plotting import figure


def run_filter(x0, p0, observations, Q, R):
    x_hat = np.zeros([len(observations)+1])
    p_hat = np.zeros([len(observations)+1])

    x_hat[0] = x0
    p_hat[0] = p0

    for i in range(1, len(observations)+1):
        # time update
        p_hat[i] = p_hat[i-1]+Q

        # measurement update
        K = p_hat[i]/(p_hat[i]+R)

        x_hat[i] = x_hat[i-1]+K*(observations[i-1]-x_hat[i-1])
        p_hat[i] = (1-K)*p_hat[i]

    return x_hat, p_hat


def update_filter(attrname, old, new):
    # Get the current slider values
    Q = process_variance.value
    R = measurement_variance.value ** 2
    x0 = initial_x.value
    p0 = initial_p.value
    x = np.arange(0, N+1)
    x_hat, p_hat = run_filter(x0, p0, observations.data['y'], Q, R)
    filter_output.data = dict(x=x, y=x_hat)


def update_data(attrname, old, new):
    # Get the current slider values
    R = measurement_variance.value

    # Generate the new curve
    x = np.arange(1, N+1)
    y = np.random.normal(true_x, R, size=N)

    observations.data = dict(x=x, y=y)
    update_filter(attrname, old, new)


# Set up data
N = 50
x = np.arange(1, N)
true_x = 0.37727
sigma = 0.1
x0 = 0.0
p0 = 0.1
Q = 1e-5
R = 0.1

y = np.random.normal(true_x, sigma, size=N)
observations = ColumnDataSource(data=dict(x=x, y=y))

x_hat, p_hat = run_filter(x0, p0, observations.data['y'], Q, R)
filter_output = ColumnDataSource(data=dict(x=x, y=x_hat))

# Set up plot
plot = figure(height=400, width=400, title="my sine wave",
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[0, N], y_range=[true_x-4*sigma, true_x+4*sigma])

plot.scatter('x', 'y', source=observations, line_width=3, line_alpha=0.6)
plot.line([0, N], [true_x, true_x], color="green")
plot.line('x', 'y', source=filter_output, color="blue")

# Set up widgets
measurement_variance = Slider(title="Measurement Variance", value=0.1, start=0.1, end=0.2, step=0.01)
process_variance = Slider(title="Process Variance", value=1e-5, start=1e-5, end=10e-5, step=1e-5)
initial_x = Slider(title="Initial x", value=0.0, start=0.0, end=1.0, step=0.1)
initial_p = Slider(title="Initial P", value=0.1, start=0.1, end=1.0, step=0.1)


measurement_variance.on_change('value', update_data)

for w in [process_variance, initial_x, initial_p]:
    w.on_change('value', update_filter)


# Set up layouts and add to document
inputs = column(measurement_variance, process_variance, initial_x, initial_p)

curdoc().add_root(row(inputs, plot, width=800))
curdoc().title = "Sliders"

#!/usr/bin/env python3
''' Present an interactive function explorer with slider widgets.

Scrub the sliders to change the properties of the filter.

Use the ``bokeh serve`` command to run the example by executing:

    python3 -m bokeh serve 1D-example.py

at your command prompt. Then navigate to the URL

    http://localhost:5006/1D-example

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
        p_hat[i] = p_hat[i-1]+Q

        # Calculate Kalman Gain
        K = p_hat[i]/(p_hat[i]+R)

        # Calculate state and variance estimate
        x_hat[i] = x_hat[i-1]+K*(observations[i-1]-x_hat[i-1])
        p_hat[i] = (1-K)*p_hat[i]

    return x_hat, p_hat


def update_filter(attrname, old, new):
    # Get the current slider values
    Q = process_variance.value * 1e-6
    R = measurement_variance.value ** 2
    N = observation_count.value
    x0 = initial_x.value
    p0 = initial_p.value
    k = np.arange(0, N+1)
    x_hat, p_hat = run_filter(x0, p0, observations.data['y'], Q, R)
    filter_output.data = dict(x=k, y=x_hat)
    variance_output.data = dict(x=k, y=3*p_hat)
    filter_error.data = dict(x=k, y=np.abs(x_hat-true_x))


def update_data(attrname, old, new):
    # Get the current slider values
    N = observation_count.value
    R = measurement_variance.value

    # Generate the new curve
    k = np.arange(1, N+1)
    y = np.random.normal(true_x, R, size=N)

    observations.data = dict(x=k, y=y)
    update_filter(attrname, old, new)


# Set up data
N = 50
k = np.arange(1, N+1)
k_hat = np.arange(0, N+1)
true_x = 0.37727
sigma = 0.1
x0 = 0.0
p0 = 0.1
Q = 0.0
R = 0.1

y = np.random.normal(true_x, sigma, size=N)
observations = ColumnDataSource(data=dict(x=k, y=y))

x_hat, p_hat = run_filter(x0, p0, observations.data['y'], Q, R)
filter_output = ColumnDataSource(data=dict(x=k_hat, y=x_hat))
variance_output = ColumnDataSource(data=dict(x=k_hat, y=3*p_hat))
filter_error = ColumnDataSource(data=dict(x=k_hat, y=np.abs(x_hat-true_x)))

# Set up plot
state_plot = figure(height=400, width=400, title='1-D Kalman Filter State',
                    tools='crosshair,pan,reset,save,wheel_zoom')
state_plot.scatter('x', 'y', source=observations, line_width=3, line_alpha=0.6)
state_plot.line([0, N], [true_x, true_x], color='green')
state_plot.line('x', 'y', source=filter_output, color='blue')

cov_plot = figure(height=400, width=400, title='1-D Kalman Filter 3-sigma Variance',
                  tools='crosshair,pan,reset,save,wheel_zoom')
cov_plot.line('x', 'y', source=variance_output, color='blue')
cov_plot.line('x', 'y', source=filter_error, color='red')


# Set up widgets
observation_count = Slider(title='Observation Count',
                           value=50, start=50, end=500, step=1)
measurement_variance = Slider(
    title='Measurement Variance', value=0.1, start=0.1, end=0.5, step=0.01)
process_variance = Slider(title='Process Variance *1e-6',
                          value=0, start=0, end=1000, step=1)
initial_x = Slider(title='Initial x', value=0.0, start=0.0, end=1.0, step=0.1)
initial_p = Slider(title='Initial P', value=0.1,
                   start=0.1, end=100.0, step=0.1)


for s in [observation_count, measurement_variance]:
    s.on_change('value', update_data)

for s in [process_variance, initial_x, initial_p]:
    s.on_change('value', update_filter)


# Set up layouts and add to document
inputs = column(observation_count, measurement_variance,
                process_variance, initial_x, initial_p)

curdoc().add_root(row(inputs, state_plot, cov_plot, width=1200))
curdoc().title = '1D-Filtering'

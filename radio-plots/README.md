Radio plots
===========

This directory contains tools to generate the plots seen in Section 3 of
the RADIO II paper. 

1. Fiesta (fiesta/)
-------------------

The Fiesta numbers were obtained on Delta AI and recorded manually in a
text file. The code to generate the plot from the recorded data is
provided.

2. Updated radio code (perf-plot/)
----------------------------------

Code is provided to both run performance tests and generate the plots.
The performance tests are run in a loop by varying relevant numbers
inside a configuration file. A server instance and a single site
instance are used. Code is also provided to plot the results.

3. Profile of the RADAR I radio code (radio-1-profile/)
-------------------------------------------------------

The profile was recorded using manually inserted timing instructions.
Due to the use of multi-processing, aggregation is not feasible while
profiling, so simple timestamps are recorded for method entry/exit with
sufficient information to distinguish between different
threads/processes. A portion of the profile is then plotted by the
provided Python script.


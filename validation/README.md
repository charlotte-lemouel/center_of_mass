# Validation of the optimal combination of kinematic and kinetic information for calculating the body center of mass

This folder contains the data and code for generating the results and figures of the following paper:

The raw data was obtained from a publicly available database (Wojtusch and von Stryk, 2015) accessed through the following link: 
https://web.sim.informatik.tu-darmstadt.de/humod2/index.html 

The code must be run in this order:
	1. preprocessing.py : the raw data is preprocessed and exported to python (this may take a while)
	2. data_analysis.py : the preprocessed data is analysed
	3. results_and_figures.py : generates the results and figures of the paper

Reference: 
Wojtusch J, von Stryk O (2015) HuMoD &#x2014; A versatile and open database for the investigation, modeling and simulation of human motion dynamics on actuation level. In: 2015 IEEE-RAS 15th International Conference on Humanoid Robots (Humanoids) , pp. 74–79. IEEE Press, Seoul, South Korea. https://doi.org/10.1109/HUMANOIDS.2015.7363534
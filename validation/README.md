# Validation of the optimal combination of kinematic and kinetic information for calculating the body center of mass

This folder contains the data and code for generating the results and figures in (Le Mouel 2024).

The raw data was obtained from two publicly available databases. 
Dataset 1 (Wojtusch and von Stryk, 2015) was accessed through the following link: 
https://web.sim.informatik.tu-darmstadt.de/humod2/index.html 
Dataset 2 (Srinivasan & Seethapathi, 2019) was accessed through the following link: 
https://doi.org/10.5061/dryad.1nt24m0 

The code must be run in this order:
1. preprocessing.py : the raw data is preprocessed and exported to python (this may take a while)
2. data_analysis.py : the preprocessed data is analysed
3. results_and_figures.py : generates the results and figures of the paper

References: 
Le Mouel, C. (2024). Optimal merging of kinematic and kinetic information to determine the position of the whole body Center of Mass. biorXiv. doi: 10.1101/2024.07.24.604923
Wojtusch J, von Stryk O (2015) HuMoD &#x2014; A versatile and open database for the investigation, modeling and simulation of human motion dynamics on actuation level. In: 2015 IEEE-RAS 15th International Conference on Humanoid Robots (Humanoids) , pp. 74–79. IEEE Press, Seoul, South Korea. https://doi.org/10.1109/HUMANOIDS.2015.7363534
Seethapathi N, Srinivasan M (2019) Step-to-step variations in human running reveal how humans run without falling (D Lentink, AJ King, A Biewener, G Sawicki, Eds,). eLife, 8, e38371. https://doi.org/10.7554/eLife.38371
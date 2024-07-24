import numpy, scipy, scipy.stats
trials    = ['2.2','2.3']
models    = ['3D-segments','Long-axis','hips']
Frequency = 500

# Visualisation
import matplotlib.pyplot as plt  
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
dimensions      = ['Lateral','Forwards','Upwards','Amplitude']
plot_dimensions = ['Upwards','Forwards','Lateral','Amplitude']
model_colors    = {'3D-segments':'g','Long-axis':'orange','hips':'r'}

# Kinetic Error: error in CoM estimation due to forceplate noise
kinetic_drift = numpy.load('data\\kinetic_drift.npz')['KineticDrift']
nb_p_stds, nb_unloaded_trials, nb_dims, unloaded_duration = numpy.shape(kinetic_drift)
kinetic_error = numpy.zeros((4,nb_p_stds)) # 
kinetic_drift_square_amplitude = numpy.sum(kinetic_drift**2, axis = 2) # summing over dimensions
for p in range(nb_p_stds):
    for d in range(3):
        kinetic_error[d,p] = numpy.sqrt(numpy.mean(kinetic_drift[p,:,d]**2)) # averaging over trials and duration
    kinetic_error[3,p] = numpy.sqrt(numpy.mean(kinetic_drift_square_amplitude[p])) # averaging over trials and duration

# Periodic Positions
import pickle
pickle_file = open('data\\periodic_positions.pkl','rb')
periodic_position = pickle.load(pickle_file)
pickle_file.close()
P_stds = numpy.array(periodic_position['P_stds'])
nb_p_stds = len(P_stds)

position_error    = {'kinematic':{},'estimate':{}}
periodic_distance = {'kinematic':{},'estimate':{}}

for model in models:
    periodic_difference = {'kinematic':[[],[],[]],'estimate':[[[] for _ in range(len(P_stds))],[[] for _ in range(len(P_stds))],[[] for _ in range(len(P_stds))]]}
    for subject in ['A','B']:
        for trial in trials:
            # Kinematics
            kinematic_difference = periodic_position['kinetic'][subject+trial]-periodic_position['kinematic'][subject+trial][model]
            for dim in range(3):
                periodic_difference['kinematic'][dim].extend(list(kinematic_difference[dim]))
            # Estimate
            for i in range(nb_p_stds):
                estimate_difference = periodic_position['kinetic'][subject+trial]-periodic_position['estimate'][subject+trial][model][i]
                for dim in range(3):
                    periodic_difference['estimate'][dim][i].extend(list(estimate_difference[dim]))
                    
    position_error['kinematic'][model] = numpy.zeros((4))
    position_error['estimate'][model]  = numpy.zeros((4, nb_p_stds))               
    for CoM_calculation in ['kinematic','estimate']:
        periodic_difference[CoM_calculation]       = numpy.array(periodic_difference[CoM_calculation])
        square_distance                            = numpy.sum(periodic_difference[CoM_calculation]**2, axis = 0) # summing over dimensions
        position_error[CoM_calculation][model][:3] = numpy.sqrt(numpy.mean(periodic_difference[CoM_calculation]**2, axis = -1)) # averaging over subjects, trials and duration
        periodic_distance[CoM_calculation][model]  = numpy.sqrt(square_distance)
        position_error[CoM_calculation][model][3]  = numpy.sqrt(numpy.mean(square_distance, axis = -1)) # averaging over subjects, trials and duration
        
kinetic_square_distance = []
for subject in ['A','B']:
    for trial in trials:
        kinetic_square_distance.extend(list(numpy.sum(periodic_position['kinetic'][subject+trial]**2, axis = 0)))
kinetic_amplitude = numpy.sqrt(numpy.mean(kinetic_square_distance))
print('Kinetic amplitude (mm): %0.1f'%(1000*kinetic_amplitude))
       
# Confidence interval : the confidence interval on position error is obtained by multiplying the lower and upper bounds by:
NbOfSamples = len(periodic_distance['kinematic'][model])
lower_bound = numpy.sqrt((NbOfSamples-1)/scipy.stats.chi2.ppf(0.975, NbOfSamples))
upper_bound = numpy.sqrt((NbOfSamples-1)/scipy.stats.chi2.ppf(0.025, NbOfSamples))

        
# p_std minimising the Total Estimation Error 
print(r'p_std minimising the total estimation error (in mm)')
position_error['total estimate'] = {}
p_std_min_index  = {} # index of P_std which minimises the amplitude of the total estimation error
amplitude_index  = dimensions.index('Amplitude')
for model in models:
    position_error['total estimate'][model] = numpy.sqrt(kinetic_error**2 + position_error['estimate'][model]**2)
    p_std_min_index[model]  = numpy.argmin(position_error['total estimate'][model][amplitude_index,:])
    print(model, 'p_std*=',1000*P_stds[p_std_min_index[model]],' Kinetic Error Amplitude= %0.2f'%(1000*kinetic_error[amplitude_index,p_std_min_index[model]]))
    
for CoM_calculation in ['kinematic','estimate','total estimate']:    
    print('')
    print('*** '+CoM_calculation+' Position Error (in mm) ***') 
    for model in models:
        print('')
        print('**',model,'**')
        for d, dimension in enumerate(plot_dimensions):
            d_index = dimensions.index(dimension)
            if CoM_calculation == 'kinematic':
                error   = 1000*position_error[CoM_calculation][model][d_index]
            else:
                error   = 1000*position_error[CoM_calculation][model][d_index,p_std_min_index[model]]
            print(dimension)
            print('%0.2f'%error)
            if dimension != 'Amplitude' and CoM_calculation != 'total estimate':
                print('(%0.2f to %0.2f)'%(error*lower_bound, error*upper_bound))
            elif dimension == 'Amplitude':
                print('(%0.1f) percent'%(100*error/(1000*kinetic_amplitude)))
print('')
print('*** Statistics on Position Error ***')
model_comparisons = [['3D-segments','Long-axis'],['Long-axis','hips'],['3D-segments','hips']]
print('')
print('Kinematics')
for small, large in model_comparisons:
    statistic, pvalue = scipy.stats.mannwhitneyu(periodic_distance['kinematic'][small], periodic_distance['kinematic'][large], use_continuity=True, alternative='less')
    print(small+' < '+large+' : p = ',pvalue,' (Mann-Whitney U)')
print('')
print('Estimator')
for small, large in model_comparisons:
    min_index_small   = p_std_min_index[small]
    min_index_large   = p_std_min_index[large]
    statistic, pvalue = scipy.stats.mannwhitneyu(periodic_distance['estimate'][small][min_index_small], periodic_distance['estimate'][large][min_index_large], use_continuity=True, alternative='less')
    print(small+' < '+large+' : p = ',pvalue,' (Mann-Whitney U)')
print('')     
for model in models:
    min_index = p_std_min_index[model]
    statistic, pvalue = scipy.stats.mannwhitneyu(periodic_distance['estimate'][model][min_index], periodic_distance['kinematic'][model], use_continuity=True, alternative='less')
    print(model, 'Estimator < Kinematics : p = ',pvalue)


# Acceleration
pickle_file = open('data\\flight_acceleration.pkl','rb')
Flight_acceleration = pickle.load(pickle_file)
pickle_file.close()

AccelerationBias         = {'kinematic':{},'estimate':{}}
# Confidence intervals
NbOfSamples_flight = numpy.shape(Flight_acceleration['kinematic']['3D-segments'])[-1]
rv                 = scipy.stats.binom(NbOfSamples_flight, 0.5)
binom_critical     = int(rv.ppf(q=0.025)) 
binom_critical     = int(rv.ppf(q=0.05)) 
Acceleration_lower_bound = {'kinematic':{},'estimate':{}}
Acceleration_upper_bound = {'kinematic':{},'estimate':{}}

for model in models:
    AccelerationBias['kinematic'][model]     = numpy.zeros((4))
    AccelerationBias['estimate'][model]      = numpy.zeros((4,nb_p_stds))
    for CoM_calculation in ['kinematic','estimate']:
        Flight_acceleration[CoM_calculation][model][2] += 9.81 # the acceleration of gravity is added to the vertical direction
        AccelerationBias[CoM_calculation][model][:3]    = numpy.median(Flight_acceleration[CoM_calculation][model],axis=-1)
        # Acceleration Bias Amplitude
        AccelerationBias[CoM_calculation][model][3]     = numpy.sqrt(numpy.sum(AccelerationBias[CoM_calculation][model][:3]**2,axis = 0))
        # Confidence interval 
        Sorted_Flight_acceleration = numpy.sort(Flight_acceleration[CoM_calculation][model], axis = -1)
        Acceleration_lower_bound[CoM_calculation][model] = (Sorted_Flight_acceleration.T[binom_critical]).T
        Acceleration_upper_bound[CoM_calculation][model] = (Sorted_Flight_acceleration.T[NbOfSamples_flight-binom_critical]).T
        
for CoM_calculation in ['kinematic','estimate']:    
    print('')
    print('*** '+CoM_calculation+' Acceleration Bias (in m/s^2) ***') 
    for model in models:
        print('')
        print('**',model,'**')
        for d, dimension in enumerate(plot_dimensions):
            d_index = dimensions.index(dimension)
            if CoM_calculation == 'kinematic':
                acceleration_bias  = AccelerationBias[CoM_calculation][model][d_index]
                if dimension != 'Amplitude':
                    acceleration_lower = Acceleration_lower_bound[CoM_calculation][model][d_index]
                    acceleration_upper = Acceleration_upper_bound[CoM_calculation][model][d_index]
            else:
                acceleration_bias  = AccelerationBias[CoM_calculation][model][d_index,p_std_min_index[model]]
                if dimension != 'Amplitude':
                    acceleration_lower = Acceleration_lower_bound[CoM_calculation][model][d_index,p_std_min_index[model]]
                    acceleration_upper = Acceleration_upper_bound[CoM_calculation][model][d_index,p_std_min_index[model]]
            print(dimension)
            print('%0.2f'%acceleration_bias)
            if dimension != 'Amplitude':
                print('(%0.2f to %0.2f)'%(acceleration_lower, acceleration_upper))
            elif dimension == 'Amplitude':
                print('(%0.1f) percent'%(100*acceleration_bias/9.81))
        

print('*** Statistics of the acceleration: is the median bias different to zero (one sample sign test)? ***')
for CoM_calculation in ['kinematic','estimate']:    
    print('')
    print('** '+CoM_calculation+' acceleration == 0 ?**') 
    for model in models:
        print('')
        print('*',model,'*')
        for d, dimension in enumerate(plot_dimensions[:3]):
            d_index = dimensions.index(dimension)
            if CoM_calculation == 'kinematic':
                acceleration_error = Flight_acceleration['kinematic'][model][d_index]
            elif CoM_calculation == 'estimate':
                acceleration_error = Flight_acceleration['estimate'][model][d_index,p_std_min_index[model]]
            acceleration_error = acceleration_error[acceleration_error != 0] # remove zero values
            n_pos = numpy.sum(acceleration_error > 0)
            n_neg = numpy.sum(acceleration_error < 0)
            # We use the smaller of n_pos and n_neg as our test statistic (for a two-tailed test)
            n = numpy.min([n_pos, n_neg])
            # Calculate p-value (two-tailed) using the binomial test
            result = scipy.stats.binomtest(n, n_pos + n_neg, p=0.5, alternative='two-sided')
            print(dimension,'p-value: %0.2g'%result.pvalue)

## Figures

# Figure 3. Improved accuracy of the estimate
fig, axes = plt.subplots(nrows = 2, ncols = len(plot_dimensions))
for d, dimension in enumerate(plot_dimensions):
    d_index = dimensions.index(dimension)
    axes[0,d].set_title(dimension)
    
    # Acceleration Bias 
    for model in models:
        estimate_bias  = AccelerationBias['estimate'][model][d_index]
        kinematic_bias = AccelerationBias['kinematic'][model][d_index]
        axes[0,d].plot(1000*P_stds,estimate_bias, color = model_colors[model])
        if d < 3:
            axes[0,d].fill_between(1000*P_stds,Acceleration_lower_bound['estimate'][model][d_index], y2 = Acceleration_upper_bound['estimate'][model][d_index], color = model_colors[model])
            kinematic_lower = Acceleration_lower_bound['kinematic'][model][d_index]
            kinematic_upper = Acceleration_upper_bound['kinematic'][model][d_index]
            kinematic_yerr  = numpy.array([[(kinematic_bias-kinematic_lower),(kinematic_upper-kinematic_bias)]]).T
            axes[0,d].errorbar(0,kinematic_bias, yerr=kinematic_yerr,fmt = '.', ecolor = model_colors[model], color = model_colors[model])#
            
        else:    
            axes[0,d].scatter(0,kinematic_bias, color = model_colors[model], marker ='.')
    if d < 3:
        axes[0,d].set_ylim(-2.8,1)
    # Position Error
    axes[1,d].plot(1000*P_stds, 1000*kinetic_error[d_index], color = 'k')
    for model in models:
        axes[1,d].plot(1000*P_stds,1000*position_error['total estimate'][model][d_index], color = model_colors[model], ls = '--')
        kinematic_error = 1000*position_error['kinematic'][model][d_index]
        estimate_error  = 1000*position_error['estimate'][model][d_index]
        axes[1,d].plot(1000*P_stds,estimate_error, color = model_colors[model])
        if d < 3:
            axes[1,d].fill_between(1000*P_stds,lower_bound*estimate_error, y2 = upper_bound*estimate_error, color = model_colors[model])
            axes[1,d].errorbar(0,kinematic_error, yerr=kinematic_error*numpy.array([[(1-lower_bound),(upper_bound-1)]]).T,fmt = '.', ecolor = model_colors[model], color = model_colors[model])#
        else:    
            axes[1,d].scatter(0,kinematic_error, color = model_colors[model], marker ='.')
    axes[1,d].set_ylim(0,12.5)
    axes[1,d].set_xlabel(r'$p_{std}   (mm)$')
axes[1,0].set_ylabel('Position Error (mm)')
axes[0,0].set_ylabel(r'Acceleration Bias $(m/s^2)$')

 
# Figure 4. Sensitivity to p_std
fig, axes = plt.subplots(ncols = 2)
for model in models:
    P_std_min = P_stds[p_std_min_index[model]]
    axes[0].plot(P_stds/P_std_min,AccelerationBias['estimate'][model][3], color = model_colors[model])
    axes[1].plot(P_stds/P_std_min,1000*position_error['total estimate'][model][3], color = model_colors[model])
axes[1].set_ylim(0,3)
axes[1].set_ylabel('Total Estimate Error Amplitude (mm)')
axes[0].set_ylim(0,1)
axes[0].set_ylabel(r'Acceleration Bias Amplitude($m/s^2$)')
for i in range(2):
    axes[i].set_xlim(0,5)
    axes[i].set_xlabel(r'$p_{std}/p_{std}^{*}$')

# Figure 1. Periodic positions
# Averaging over subjects and trials as a function of % gait cycle
AverageCoM = {}
AverageCoM['kinetic']  = numpy.zeros((3,100))
AverageCoM['kinematic'] = {}
AverageCoM['estimate']  = {}
for model in models:
    AverageCoM['kinematic'][model] = numpy.zeros((3,100))
    AverageCoM['estimate'][model]  = numpy.zeros((3,100))
InterpolatedTime = numpy.arange(100)

for subject in ['A','B']:
    for trial in trials:
        kinetic   = periodic_position['kinetic'][subject+trial]
        kinematic = periodic_position['kinematic'][subject+trial]
        estimate  = periodic_position['estimate'][subject+trial]
        NbOfSamples      = numpy.shape(kinetic)[1]
        NormalisedTime   = 100*numpy.arange(NbOfSamples)/NbOfSamples
        for dim in range(3):
            AverageCoM['kinetic'][dim] += 0.25*(numpy.interp(InterpolatedTime, NormalisedTime, kinetic[dim]))
            for model in models:
                AverageCoM['kinematic'][model][dim] += 0.25*(numpy.interp(InterpolatedTime, NormalisedTime, kinematic[model][dim]))
                AverageCoM['estimate'][model][dim]  += 0.25*(numpy.interp(InterpolatedTime, NormalisedTime, estimate[model][p_std_min_index[model],dim]))
                
fig, axes = plt.subplots(nrows = 2, ncols = 3)
for d, dimension in enumerate(plot_dimensions[:3]):
    d_index = dimensions.index(dimension)
    # kinetic
    for row in range(2):  
        axes[row,d].plot(InterpolatedTime, 1000*AverageCoM['kinetic'][d_index], color = 'k')
    for model in models:
        # kinematic
        axes[0,d].plot(InterpolatedTime, 1000*AverageCoM['kinematic'][model][d_index], color = model_colors[model], label = model)
        # estimate
        axes[1,d].plot(InterpolatedTime, 1000*AverageCoM['estimate'][model][d_index], color = model_colors[model], label = model, ls = ':')
    axes[0,d].set_title(dimension)
    axes[1,d].set_xlabel('Time (% gait cycle)')
for row in range(2):    
    axes[row,2].set_yticks([])
    axes[row,0].set_ylabel('Position (mm)')    
    for col in range(3):
        axes[row,col].set_ylim(-35,35)
           

# Supplementary Figure 1. Periodic positions of individual trials
fig, axes = plt.subplots(nrows = 2*len(trials),ncols = 3)
for s, subject in enumerate(['A','B']):
    for t, trial in enumerate(trials):
        kinetic   = periodic_position['kinetic'][subject+trial]
        kinematic = periodic_position['kinematic'][subject+trial]
        NbOfSamples      = numpy.shape(kinetic)[1]
        time             = numpy.arange(-int(NbOfSamples/2),int(NbOfSamples/2))/Frequency
        for d, dimension in enumerate(plot_dimensions[:3]):
            d_index = dimensions.index(dimension)
            axes[2*t+s,d].plot(time,1000*kinetic[d_index], color = 'k')
            for model in models:
                axes[2*t+s,d].plot(time,1000*kinematic[model][d_index], model_colors[model], label = model)
        # axes[2*t+s,0].set_ylabel(subject+trial)
for d, dimension in enumerate(plot_dimensions[:3]):
    d_index = dimensions.index(dimension)     
    axes[0,d].set_title(dimension)
    for row in range(2*len(trials)):
        axes[row,d].set_ylim(-41,41)
        axes[row,d].set_xlim(-0.4,0.4)
        if row < 2*len(trials)-1:
            axes[row,d].set_xticks([])
        else:
            axes[row,d].set_xlabel('Time (s)')
        if d > 0:
            axes[row,d].set_yticks([])
        else:
            axes[row,d].set_ylabel('Position (mm)')

# Supplementary Figure 2. Kinetic Error
P_stds_visualise = [0.0002, 0.002, 0.0195]
time       = numpy.arange(unloaded_duration)/Frequency
kinetic_drift_amplitude = numpy.sqrt(kinetic_drift_square_amplitude)

# Comparison to double integration
Weight    = numpy.load('data\weight.npz')
Unloaded_force    = numpy.load('data\\unloaded_force.npz')['Unloaded_force']
integration_drift = numpy.zeros((nb_unloaded_trials,3,unloaded_duration))
for s, subject in enumerate(['A','B']):
    Mass = Weight[subject]/9.81
    for segment in range(int(nb_unloaded_trials/2)):
        Acceleration = Unloaded_force[segment]/Mass
        Velocity     = numpy.cumsum(Acceleration, axis = 1)/Frequency
        integration_drift[int(nb_unloaded_trials/2)*s+segment] = numpy.cumsum(Velocity, axis = 1)/Frequency
integration_drift_amplitude = numpy.sqrt(numpy.sum(integration_drift**2, axis = 1)) # summing over dimensions
        
fig, axes  = plt.subplots(len(P_stds_visualise)+1)
# Kinetic Error
for p, p_std in enumerate(P_stds_visualise):
    p_index = numpy.argmax(P_stds>=p_std)
    for trial in range(nb_unloaded_trials):
        axes[p].plot(time, 1000*kinetic_drift_amplitude[p_index,trial], lw = 0.5)
    axes[p].set_title(r'$p_{std}$ = %0.2g mm, Kinetic Error = %0.2g mm'%(1000*p_std,1000*kinetic_error[3,p_index]))
    axes[p].set_xticks([])
# Double Integration Error
for trial in range(nb_unloaded_trials):
    axes[len(P_stds_visualise)].plot(time, 1000*integration_drift_amplitude[trial], lw = 0.5)
axes[len(P_stds_visualise)].set_title(r'Double integration error = %i mm'%(1000*numpy.sqrt(numpy.mean(integration_drift_amplitude**2))))
axes[len(P_stds_visualise)].set_xlabel('Time (s)')
axes[len(P_stds_visualise)].set_xlabel('Time (s)')
for p in range(len(P_stds_visualise)+1):
    axes[p].set_ylabel('Amplitude (mm)')


# Figure 2. Transfer function
def estimator_gains(ratio):
    l2    = (4*ratio+1 - numpy.sqrt(1+8*ratio))/(4*ratio**2)
    l1    = 1 - ratio**2*l2**2
    return l1, l2
    
Powers = numpy.arange(-3,6)
Ratios = 10.**Powers
l1s, l2s = estimator_gains(Ratios)

import matplotlib
cmap = matplotlib.cm.get_cmap('plasma') #coolwarm viridis
norm = matplotlib.colors.Normalize(vmin=-6, vmax=6)
labels = []
colors = []
for power in numpy.arange(-3,6) :
    labels.append(r'$10^{%i}$'%int(power))
    colors.append(cmap(norm(power)))
    
Freq_normed = numpy.arange(0,0.5,0.001)
z = numpy.array([numpy.exp(1j*2*numpy.pi*Freq_normed)]).T

Denominator = z**2-z*(2-l1s-l2s)+1-l1s
Position_transfer = (l1s*(z-1)+l2s)/Denominator
Acceleration_transfer_ratio = (1-l1s)*(z-1)**2/Denominator

fig, axes = plt.subplots(ncols = 3)
for s, signal in enumerate([Position_transfer, Acceleration_transfer_ratio]):
    for r in range(len(Ratios)):
        axes[s].plot(Freq_normed, abs(signal[:,r]), color = colors[r], label = labels[r])
    axes[s].set_xlabel('Frequency/F')
    axes[s].set_ylim(0,1.3)
for r in range(len(Ratios)):
    axes[2].plot([0,0],[0,1],color = colors[r], label = labels[r])
axes[2].legend()
axes[0].set_ylabel('Transfer function amplitude')

# Appendix B, Figure 1. Estimator gains
Powers = numpy.arange(-3,6,0.1)
Ratios = 10.**Powers
l1s, l2s = estimator_gains(Ratios)
fig, ax = plt.subplots()
ax.plot(Powers, l1s, label = r'$l_1$')
ax.plot(Powers, l2s, label = r'$l_2$')
ax.legend()
ax.set_xticks(numpy.arange(-3,6), labels)
ax.set_xlabel(r'$r=\frac{p_{std}}{a_{std}T^2}$')

plt.show()
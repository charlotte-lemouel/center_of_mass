import numpy, scipy.io

# Function to read in a MAT file and then convert it to an easily accessible 
# dictionary. These functions come thanks to the post:
# https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
def loadmat(filename):
    '''
    this function should be called instead of direct scipy.io.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], scipy.io.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, numpy.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, scipy.io.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, numpy.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)
    
def gap_filling(position, visibility):
    ''' Function for filling in gaps in marker data through linear interpolation
    
    Parameters
    ----------
    position: (dict of str : (3, NbOfSamples) numpy.ndarray)
        Dictionary of 'marker name': 3D trajectory of the marker
    visibility: (dict of str : (NbOfSamples,) numpy.ndarray)
        Dictionary of 'marker name': array with value 1 when the marker is visible, 0 when the marker is invisible
        
    Returns
    -------
    new_position: (dict of str : (3, NbOfSamples) numpy.ndarray)
        Dictionary of 'marker name': gap-filled 3D trajectory of the marker
    '''
    duration        = numpy.shape(position)[1]
    new_position    = numpy.zeros((3,duration))
    new_visibility  = numpy.ones((duration))
    new_visibility[visibility==0] = 0
    new_position   += position
    
    if numpy.sum(visibility) == 0:
        print('invisible')
        return position
        
    else:
        ## Beginning
        if visibility[0] == 0:
            t = numpy.argmax(visibility)
            new_position[:,:t] = position[:,t].repeat(t).reshape(3,t)
            new_visibility[:t] = numpy.ones(t)

        ## End
        if visibility[-1] == 0:
            reversed = new_visibility[::-1]
            t = numpy.argmax(reversed)
            new_position[:,-t:] = position[:,-t-1].repeat(t).reshape(3,t)
            new_visibility[-t:] = numpy.ones(t)	


        while numpy.mean(new_visibility) < 1:
            # Find the first hole
            tmin = numpy.argmin(new_visibility)
            tmax = tmin + numpy.argmax(new_visibility[tmin:])
            # Fill it in
            new_position[:,tmin:tmax] = numpy.array([position[:,tmin-1]]).T + numpy.array([position[:,tmax] - position[:,tmin-1]]).T*numpy.array([numpy.arange(1,tmax - tmin + 1)])/(tmax - tmin + 1.)
            new_visibility[tmin:tmax] = numpy.ones(tmax-tmin)
        return new_position


def flight_values(signal3D, Flight_onsets, Stance_onsets):
    '''Function to extract the values of a signal during flight 
    '''
    Flight_values = [[],[],[]]
    for f, flight_onset in enumerate(Flight_onsets):
        for dim in range(3):
            Flight_values[dim].extend(list(signal3D[dim,flight_onset:Stance_onsets[f]]))
    return Flight_values        

## Functions for calculating the averages across steps
def flip_lateral(all_steps, initial_side, lateral = 0):
    '''For running, flip the lateral signal for all the Left steps
    '''
    if initial_side == 'Left':
        flip_start = 0
    elif initial_side == 'Right':
        flip_start = 1
    NbSteps = numpy.shape(all_steps)[0]
    all_steps_flip = numpy.copy(all_steps)
    for i in range(flip_start,NbSteps,2):
        all_steps_flip[i,lateral] = - all_steps_flip[i,lateral]
    return all_steps_flip
    
def average_across_steps(signal3D, indices, trial, initial_side, pre_samples, post_samples = None):
    if post_samples == None:
        post_samples = pre_samples
    all_steps = numpy.zeros((len(indices), 3, pre_samples + post_samples))
    for o, onset in enumerate(indices): 
        all_steps[o] = signal3D[:,onset - pre_samples : onset + post_samples]  
    if trial.startswith('2'): #running trials
        all_steps = flip_lateral(all_steps, initial_side)
    return numpy.mean(all_steps, axis = 0) # averaging across steps   
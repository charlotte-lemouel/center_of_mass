function Acceleration = com_acceleration(GroundReactionForce, mass, gravity_direction)
% Acceleration = com_acceleration(GroundReactionForce, mass, gravity_direction)
%
% Calculates the CoM acceleration from the Ground reaction force and mass 
%
% Parameters
% ----------
% GroundReactionForce: (NbOfDimensions,NbOfSamples) double array
% 	Ground reaction force (in Newton)
% mass: float 
% 	subject's mass (in kg)
% gravity_direction: (NbOfDimensions) double array, optional
% 	direction of the gravity vector used to subtract the subject's weight, default is [0,0,-1]
%
% Returns
% -------
% Acceleration: (NbOfDimensions,NbOfSamples) double array
% 	Acceleration of the Center of Mass (in m/s^2)

if nargin < 3
    gravity_direction = [0,0,-1]; % Default value if gravity_direction is not provided
end

% The net force is the sum of the ground reaction force and the person's weight 
NetForce = GroundReactionForce + repmat(mass*9.81*gravity_direction, length(GroundReactionForce),1)';
% Acceleration is force divided by mass
Acceleration = NetForce/mass; % in m/s^2

end
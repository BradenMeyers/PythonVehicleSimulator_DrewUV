import numpy as np

class fin:
    '''
    Represents a fin for hydrodynamic calculations.

    INPUTS:
        a:      fin area (m^2)
        CL:     coefficient of lift (dimensionless)
        x:      x distance (m) from center of vehicle (negative for behind COM)
        c:      radius (m) from COP on the fin to the COM in the YZ plane
        angle:  offset of fin angle around x axis (deg) starting from positive y
                0 deg: fin on left side looking from front
                90 deg: fin on bottom
        cs:     control subsystem the fin is going to actuate for
        rho:    density of fluid (kg/m^3) 
    
    Coordinate system: Right-handed, x-forward, y-starboard, z-down
    '''
    
    def __init__(
            self,
            a,
            CL,
            x,
            c = 0,
            angle = 0,
            rho = 1026,  # Default density of seawater
    ):
        
        self.area = a  # Fin area (m^2)
        self.CL = CL   # Coefficient of lift (dimensionless)
        self.angle_rad = np.deg2rad(angle)  # Convert angle to radians
        self.rho = rho  # Fluid density (kg/m^3)

        self.fin_actual = 0.0  #Actual position of the fin (rad)
        self.T_delta = 0.1              # fin time constant (s) 
        self.deltaMax = np.deg2rad(15) # max rudder angle (rad)

        # Calculate fin's Center of Pressure (COP) position relative to COM
        y = np.cos(self.angle_rad) * c  # y-component of COP (m)
        z = np.sin(self.angle_rad) * c  # z-component of COP (m)
        self.R = np.array([x, y, z])    # Location of COP of the fin relative to COM (m)

    def velocity_in_rotated_plane(self, nu_r):
        """
        Calculate velocity magnitude in a plane rotated around the x-axis.

        Parameters:
            nu_r (numpy array): Velocity vector [vx, vy, vz] (m/s) in ENU frame

        Returns:
            float: Magnitude of velocity (m/s) in the rotated plane.
        """
        # Extract velocity components
        vx, vy, vz = nu_r  # m/s

        # Rotate y component around x-axis to align with fin plane
        vy_rot = np.sqrt((vy * np.sin(self.angle_rad))**2 + (vz * np.cos(self.angle_rad))**2)

        # Calculate magnitude in the rotated plane (x, y')
        U_plane = np.sqrt(vx**2 + vy_rot**2)

        return U_plane  # m/s

    def torque(self, Ur):
        """
        Calculate torque generated by the fin.

        Parameters:
            Ur (numpy array): Relative velocity [vx, vy, vz, p, q, r] 
                              (m/s for linear, rad/s for angular)
            delta (float): Deflection angle of fin in radians
                           (positive: CW rotation of fin and trailing edge) 

        Returns:
            numpy array: tau vector [Fx, Fy, Fz, Tx, Ty, Tz] (N*m) in body-fixed frame
        """
        
        ur = self.velocity_in_rotated_plane(Ur[:3])  # Calulate relative velocity in plane of the fin
        
        # Calculate lift force magnitude
        f = 0.5 * self.rho * self.area * self.CL * self.fin_actual * ur**2  # N

        # Decompose force into y and z components
        fy = np.sin(self.angle_rad) * f  # N
        fz = -np.cos(self.angle_rad) * f  # N 

        F = np.array([0, fy, fz])  # Force vector (N)

        # Calculate torque using cross product of force and moment arm
        torque = np.cross(self.R, F)  # N*m
        return np.append(F, torque)
    
    def actuate(self, sampleTime, command):
        # Actuator dynamics        
        delta_dot = (command - self.fin_actual) / self.T_delta  
        self.fin_actual += sampleTime * delta_dot  # Euler integration 

        # Amplitude Saturation
        if abs(self.fin_actual) >= self.deltaMax:
            self.fin_actual = np.sign(self.fin_actual) * self.deltaMax

        return self.fin_actual

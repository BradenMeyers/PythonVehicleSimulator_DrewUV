#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
drewUV.py:  

   Class for the Remus 100 cylinder-shaped autonomous underwater vehicle (AUV), 
   which is controlled using a tail rudder, stern planes and a propeller. The 
   length of the AUV is 1.6 m, the cylinder diameter is 19 cm and the 
   mass of the vehicle is 31.9 kg. The maximum speed of 2.5 m/s is obtained 
   when the propeller runs at 1525 rpm in zero currents.
       
  Drew UV()                           
       Step input, Right and Left Elevators, rudder and propeller revolution     
   
    remus100('depthHeadingAutopilot',z_d,psi_d,n_d,V_c,beta_c)
        z_d:    desired depth (m), positive downwards
        psi_d:  desired yaw angle (deg)
        n_d:    desired propeller revolution (rpm)
        V_c:    current speed (m/s)
        beta_c: current direction (deg)                  

Methods:
        
    [nu,u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime ) returns 
        nu[k+1] and u_actual[k+1] using Euler's method. The control input is:

            u_control = [ delta_r   rudder angle (rad)
                         delta_s    stern plane angle (rad)
                         n          propeller revolution (rpm) ]

    u = depthHeadingAutopilot(eta,nu,sampleTime) 
        Simultaneously control of depth and heading using two controllers of 
        PID type. Propeller rpm is given as a step command.
       
    u = stepInput(t) generates tail rudder, elevator fins and RPM step inputs.   
       
References: 
    
    B. Allen, W. S. Vorus and T. Prestero, "Propulsion system performance 
         enhancements on REMUS AUVs," OCEANS 2000 MTS/IEEE Conference and 
         Exhibition. Conference Proceedings, 2000, pp. 1869-1873 vol.3, 
         doi: 10.1109/OCEANS.2000.882209.    
    T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and Motion 
         Control. 2nd. Edition, Wiley. URL: www.fossen.biz/wiley            

Author:     Thor I. Fossen
"""
import numpy as np
import math
import sys
from lib.control import PIDpolePlacement
from lib.gnc import crossFlowDrag,forceLiftDrag,Hmtrx,m2c,gvect,ssa

# Class Vehicle
class drewUV:
    """
    remus100()
        Rudder angle, stern plane and propeller revolution step inputs
        
    remus100('depthHeadingAutopilot',z_d,psi_d,n_d,V_c,beta_c) 
        Depth and heading autopilots
        
    Inputs:
        z_d:    desired depth, positive downwards (m)
        psi_d:  desired heading angle (deg)
        n_d:    desired propeller revolution (rpm)
        V_c:    current speed (m/s)                     Is this like the ocean current speed? I think so. #CHECK
        beta_c: current direction (deg)
    """

    def __init__(
        self,
        controlSystem="stepInput",
        r_z = 0,  
        r_psi = 0,
        r_rpm = 0,
        V_current = 0,
        beta_current = 0,
    ):

        # Constants
        self.D2R = math.pi / 180        # deg2rad
        self.rho = 1026                 # density of water (kg/m^3)
        g = 9.81                        # acceleration of gravity (m/s^2)
        
        if controlSystem == "depthHeadingAutopilot":
            self.controlDescription = (
                "Depth and heading autopilots, z_d = "
                + str(r_z) 
                + ", psi_d = " 
                + str(r_psi) 
                + " deg"
                )
            print("Autopilot Control")

        else:
            self.controlDescription = (
                "Step inputs for elevators, rudder and propeller")
            controlSystem = "stepInput"
            print("step Input control")
            
        self.ref_z = r_z
        self.ref_psi = r_psi
        self.ref_n = r_rpm
        self.V_c = V_current
        self.beta_c = beta_current * self.D2R
        self.controlMode = controlSystem
        
        # Initialize the AUV model 
        self.name = (
            "DrewUV cylinder-shaped AUV (see 'drewUV.py' for more details)")
        self.L = 1.05                # length (m)
        self.diam = 0.057*2          # cylinder diameter (m)
        
        self.nu = np.array([0, 0, 0, 0, 0, 0], float) # inital velocity vector
        self.u_actual = np.array([0, 0, 0, 0], float)    # initial control surfaces vector
        
        self.controls = [
            "Tail rudder (rad)",
            "Left Elevator (rad)",
            "Right Elevator (rad)",
            "Propeller revolution (rpm)"
            ]
        self.dimU = len(self.controls) 
        

        # Actuator dynamics
        self.deltaMax_r = 30 * self.D2R # max rudder angle (rad)
        self.deltaMax_re = 30 * self.D2R # max right elevator angle (rad)
        self.deltaMax_le = 30 * self.D2R # max left elevator plane angle (rad)
        self.nMax = 1525                # max propeller revolution (rpm)    
        self.T_delta = 0.09*3             # 0.09 second for 60 degrees acrcoding to amazon rudder/stern plane time constant (s) How many seconds to move one radian
        self.T_n = 1.0                  # propeller time constant (s)           #TODO:look at this? is this how fast they can move  
        
        if r_rpm < 0.0 or r_rpm > self.nMax:
            sys.exit("The RPM value should be in the interval 0-%s", (self.nMax))
        
        if r_z > 100.0 or r_z < 0.0:
            sys.exit('desired depth must be between 0-100 m')    
        
        # Hydrodynamics (Fossen 2021, Section 8.4.2)    
        self.S = 0.7 * self.L * self.diam    # S = 70% of rectangle L * diam
        a = self.L/2            #Distance from center to fin             # semi-axes
        b = self.diam/2   #Radius of vehicle  
        c = 1.7/2        #TODO: Know where the center of lift force is     
        self.r_bg = np.array([0, 0, 0.005], float)    # CG w.r.t. to the CO  #TODO: Find offset in meters maybe?
        self.r_bb = np.array([0, 0, 0], float)       # CB w.r.t. to the CO  #The origin of the vehicle is at the CENTER OF BOYANCY 

        # Parasitic drag coefficient CD_0, i.e. zero lift and alpha = 0
        # F_drag = 0.5 * rho * Cd * (pi * b^2)   
        # F_drag = 0.5 * rho * CD_0 * S
        Cd = 0.42                              # from Allen et al. (2000)  TODO: FIND coefficient of DRAG
        self.CD_0 = Cd * math.pi * b**2 / self.S
        
        #TODO: I think all of these need to be changed
        # Rigid-body mass matrix expressed in CO
        m = 7.54   #mass on october 31 TODO: Update          #4/3 * math.pi * self.rho * a * b**2     # mass of spheriod 
        Ix = (2/5) * m * b**2                       # moment of inertia
        Iy = (1/5) * m * (a**2 + b**2)
        Iz = Iy
        MRB_CG = np.diag([ m, m, m, Ix, Iy, Iz ])   # MRB expressed in the CG     
        H_rg = Hmtrx(self.r_bg)
        self.MRB = H_rg.T @ MRB_CG @ H_rg           # MRB expressed in the CO

        # Weight and buoyancy
        self.W = m * g
        self.B = self.W  #CHEKC: Should I add a little bit to account for the slight offset in boyancy
        
        # Added moment of inertia in roll: A44 = r44 * Ix
        r44 = 0.3      #CHECK: Is this the added momement due to water?       
        MA_44 = r44 * Ix
        
        # Lamb's k-factors
        e = math.sqrt( 1-(b/a)**2 )
        alpha_0 = ( 2 * (1-e**2)/pow(e,3) ) * ( 0.5 * math.log( (1+e)/(1-e) ) - e )  
        beta_0  = 1/(e**2) - (1-e**2) / (2*pow(e,3)) * math.log( (1+e)/(1-e) )

        k1 = alpha_0 / (2 - alpha_0)
        k2 = beta_0  / (2 - beta_0)
        k_prime = pow(e,4) * (beta_0-alpha_0) / ( 
            (2-e**2) * ( 2*e**2 - (2-e**2) * (beta_0-alpha_0) ) )   

        # Added mass system matrix expressed in the CO
        self.MA = np.diag([ m*k1, m*k2, m*k2, MA_44, k_prime*Iy, k_prime*Iy ])
          
        # Mass matrix including added mass
        self.M = self.MRB + self.MA
        self.Minv = np.linalg.inv(self.M)

        # Natural frequencies in roll and pitch                                  #LOOK INTO THIS! What does this mean?
        self.w_roll = math.sqrt( self.W * ( self.r_bg[2]-self.r_bb[2] ) / 
            self.M[3][3] )
        self.w_pitch = math.sqrt( self.W * ( self.r_bg[2]-self.r_bb[2] ) / 
            self.M[4][4] )
            
        # Tail rudder parameters (single)
        self.CL_delta_r = 0.5       # rudder lift coefficient   #CHECK: FIND THESE
        self.A_r = 0.10 * 0.0469  # rudder area (m2)  #I took out the times 2 because there is only one fin  #COEF
        self.x_r = -a               # rudder x-position (m) How far back it is. 
        self.z_r = -c                #rudder z-position how high up it is 

        # Right Elveator paramaters (double)  #TODO: Check Coefficients
        self.CL_delta_re = 0.7       # elevator lift coefficient   
        self.A_re = 0.10 * 0.0469  # elevator area (m2)
        self.x_re = -a               # elevator x-position (m) How far back the fin is
        self.yz_re = c              #How far the fin is from the center in yz plane

        # Left Elveator paramaters (double)
        self.CL_delta_le = 0.7       # elevator lift coefficient
        self.A_le = 0.10 * 0.0469  # elevator area (m2)
        self.x_le = -a               # elevator x-position (m) #CHECK THIS: I believe that they only care about how far back on the vehicle
        self.yz_le = -c              #How far the fin is from the center in yz plane

        # Low-speed linear damping matrix parameters
        self.T_surge = 20           # time constant in surge (s)
        self.T_sway = 20            # time constant in sway (s)
        self.T_heave = self.T_sway  # equal for for a cylinder-shaped AUV
        self.zeta_roll = 0.3        # relative damping ratio in roll
        self.zeta_pitch = 0.8       # relative damping ratio in pitch
        self.T_yaw = 5              # time constant in yaw (s)
        
        # Heading autopilot
        self.wn_psi = 0.5           # PID pole placement parameters
        self.zeta_psi = 1
        self.r_max = 1 * math.pi / 180  # maximum yaw rate 
        self.psi_d = 0                  # position, velocity and acc. states
        self.r_d = 0
        self.a_d = 0
        self.wn_d = self.wn_psi / 5     # desired natural frequency
        self.zeta_d = 1                 # desired realtive damping ratio 
        
        self.e_psi_int = 0     # yaw angle error integral state
        
        
        # Depth autopilot
        self.wn_d_z = 1/20     # desired natural frequency, reference model
        self.Kp_z = 0.1        # heave proportional gain, outer loop
        self.T_z = 100.0       # heave integral gain, outer loop
        self.Kp_theta = 1.0    # pitch PID controller     
        self.Kd_theta = 3.0  
        self.Ki_theta = 0.1

        self.z_int = 0         # heave position integral state
        self.z_d = 0           # desired position, LP filter initial state
        self.theta_int = 0     # pitch angle integral state
        

    def dynamics(self, eta, nu, u_actual, u_control, sampleTime):
        """
        [nu,u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime) integrates
        the AUV equations of motion using Euler's method.

        nu is the 
        """

        # Current velocities
        u_c = self.V_c * math.cos(self.beta_c - eta[5])  # current surge velocity  using the parameter of direction of current
        v_c = self.V_c * math.sin(self.beta_c - eta[5])  # current sway velocity

        nu_c = np.array([u_c, v_c, 0, 0, 0, 0], float) # current velocity 
        Dnu_c = np.array([nu[5]*v_c, -nu[5]*u_c, 0, 0, 0, 0],float) # derivative
        nu_r = nu - nu_c                               # relative velocity        
        alpha = math.atan2( nu_r[2], nu_r[0] )         # angle of attack 
        U = math.sqrt(nu[0]**2 + nu[1]**2 + nu[2]**2)  # vehicle speed
        U_r = math.sqrt(nu_r[0]**2 + nu_r[1]**2 + nu_r[2]**2)  # relative speed

        # Commands and actual control signals
        delta_r_c = u_control[0]    # commanded tail rudder (rad)
        delta_re_c = u_control[1]    # commanded right elevator (rad)
        delta_le_c = u_control[2]    # commanded left elevator (rad)
        n_c = u_control[3]          # commanded propeller revolution (rpm)
        
        delta_r = u_actual[0]       # actual tail rudder (rad)
        delta_re = u_actual[1]       # actual right elevator (rad)
        delta_le = u_actual[2]       # actual left elevator (rad)
        n = u_actual[3]             # actual propeller revolution (rpm)
        
        # Amplitude saturation of the control signals
        if abs(delta_r) >= self.deltaMax_r:
            delta_r = np.sign(delta_r) * self.deltaMax_r
            
        if abs(delta_re) >= self.deltaMax_re:
            delta_re = np.sign(delta_re) * self.deltaMax_re          
        
        if abs(delta_le) >= self.deltaMax_le:
            delta_le = np.sign(delta_le) * self.deltaMax_le          
            
        if abs(n) >= self.nMax:
            n = np.sign(n) * self.nMax       
        
        # Propeller coeffs. KT and KQ are computed as a function of advance no.
        # Ja = Va/(n*D_prop) where Va = (1-w)*U = 0.944 * U; Allen et al. (2000)
        D_prop = 0.076   # Blue roboitcspropeller diameter corresponding to 5.5 inches
        t_prop = 0.1    # thrust deduction number
        n_rps = n / 60  # propeller revolution (rps) 
        Va = 0.944 * U  # advance speed (m/s)

        # Ja_max = 0.944 * 2.5 / (0.14 * 1525/60) = 0.6632
        Ja_max = 0.6632
        
        # Single-screw propeller with 3 blades and blade-area ratio = 0.718.
        # Coffes. are computed using the Matlab MSS toolbox:     
        # >> [KT_0, KQ_0] = wageningen(0,1,0.718,3)
        KT_0 = 0.4566
        KQ_0 = 0.0700
        # >> [KT_max, KQ_max] = wageningen(0.6632,1,0.718,3) 
        KT_max = 0.1798
        KQ_max = 0.0312
        
        # Propeller thrust and propeller-induced roll moment
        # Linear approximations for positive Ja values
        # KT ~= KT_0 + (KT_max-KT_0)/Ja_max * Ja   
        # KQ ~= KQ_0 + (KQ_max-KQ_0)/Ja_max * Ja  
      
        #X_prop is the thrust
        #K_prop is the torque of the thruster

        if n_rps > 0:   # forward thrust

            X_prop = self.rho * pow(D_prop,4) * ( 
                KT_0 * abs(n_rps) * n_rps + (KT_max-KT_0)/Ja_max * 
                (Va/D_prop) * abs(n_rps) )        
            K_prop = self.rho * pow(D_prop,5) * (
                KQ_0 * abs(n_rps) * n_rps + (KQ_max-KQ_0)/Ja_max * 
                (Va/D_prop) * abs(n_rps) )           
            
        else:    # reverse thrust (braking)
        
            X_prop = self.rho * pow(D_prop,4) * KT_0 * abs(n_rps) * n_rps 
            K_prop = self.rho * pow(D_prop,5) * KQ_0 * abs(n_rps) * n_rps 
        
        # Rigi-body/added mass Coriolis/centripetal matrices expressed in the CO
        CRB = m2c(self.MRB, nu_r)
        CA  = m2c(self.MA, nu_r)
               
        # Nonlinear quadratic velocity terms in pitch and yaw (Munk moments) 
        # are set to zero since only linear damping is used  
        CA[4][0] = 0  
        CA[4][3] = 0
        CA[5][0] = 0
        CA[5][1] = 0
        
        C = CRB + CA

        # Dissipative forces and moments
        D = np.diag([
            self.M[0][0] / self.T_surge,
            self.M[1][1] / self.T_sway,
            self.M[2][2] / self.T_heave,
            self.M[3][3] * 2 * self.zeta_roll  * self.w_roll,
            self.M[4][4] * 2 * self.zeta_pitch * self.w_pitch,
            self.M[5][5] / self.T_yaw
            ])
        
        D[0][0] = D[0][0] * math.exp(-3*U_r) # For DOF 1,2,6 the D elements 
        D[1][1] = D[1][1] * math.exp(-3*U_r) # go to zero at higher speeds, i.e.
        D[5][5] = D[5][5] * math.exp(-3*U_r) # drag and lift/drag dominate

        tau_liftdrag = forceLiftDrag(self.diam,self.S,self.CD_0,alpha,U_r)   #calculate lift and drag from cross sectional area, and position and speed
        tau_crossflow = crossFlowDrag(self.L,self.diam,self.diam,nu_r)

        # Restoring forces and moments
        g = gvect(self.W,self.B,eta[4],eta[3],self.r_bg,self.r_bb)
        
        # Horizontal- and vertical-plane relative speed   #What is this relative speed thing
        #Check:
        U_rh = math.sqrt( nu_r[0]**2 + nu_r[1]**2 )
        U_re = math.sqrt(nu_r[0]**2 + (nu_r[1] * math.sin(self.D2R * 30))**2 + (nu_r[2] * math.sin(self.D2R * 60))**2)  #Relative speed in the plane of the elevator fin
        # U_rv = math.sqrt( nu_r[0]**2 + nu_r[2]**2 ) 

        #lift forces on the elevator fins on right and left both positve set direction below
        fl_re = 0.5 * self.rho * U_re**2 * self.A_r * self.CL_delta_r * delta_re
        fl_le = 0.5 * self.rho * U_re**2 * self.A_r * self.CL_delta_r * delta_le

        # Rudder and elevator drag # These should be the same as 4 fin because it is moving in x direction just need half of this I believe
        X_r = -0.5 * self.rho * U_rh**2 * self.A_r * self.CL_delta_r * delta_r**2  #Using the relative speed in the horziontal plane 
        X_re = -0.5 * self.rho * U_re**2 * self.A_re * self.CL_delta_re * delta_re**2    
        X_le = -0.5 * self.rho * U_re**2 * self.A_le * self.CL_delta_le * delta_le**2  
        fx = X_r + X_re + X_le

        # Rudder and elevator sway force 
        Y_r = -0.5 * self.rho * U_rh**2 * self.A_r * self.CL_delta_r * delta_r
        Y_re = -fl_re * math.sin(30 * self.D2R)
        Y_le = fl_le * math.sin(30 * self.D2R)  #Check to make sure the negative one is on the right
        fy = Y_r + Y_re + Y_le        

        # elevator heave force 
        Z_re = fl_re * math.sin(60 * self.D2R)     
        Z_le = fl_le * math.sin(60 * self.D2R)

        Mx = (Y_r * self.z_r * -1) + (self.yz_re * fl_re) + (self.yz_le * fl_le)   #CHECK: See which way it spins with the deflection of each fin
        My = (self.x_re * -Z_re) + (self.x_le * -Z_le)                            #CHECK: I did this right tilt elevators up and push vehicle forward it should pitch up or positive y moment
        Mz =  (self.x_r * Y_r) + (self.x_re * Y_re) + (self.x_le * Y_le)    #the rudder cause biggest yaw moment but rudders can too but they can cancel out

        # Generalized force vector  #TODO: Fix the force vector with X_re and Z_re
        #Looks like the vector is in format [fx, fy, fz, MX, MY, MZ] This is in the body grame
        tau = np.array([
            (1-t_prop) * X_prop + fx,  #The x forces should be the same. #TODO make the forces half as much because there is only one fin
            fy, 
            Z_re + Z_le,
            (K_prop /5) + Mx,   # scaled down by a factor of 10 to match exp. results  #TODO: Find Diameter of the force acting on fin in y-z plane to get roll moment from fin.
            My,     
            Mz     #the x_r is negative because the fin is in the negative x from center. A negative Y force causes postive Z moment!
            ], float)
    
        # AUV dynamics 
        #Tau_sum is the result of all the dynamics of the vehicle 
        tau_sum = tau + tau_liftdrag + tau_crossflow - np.matmul(C+D,nu_r)  - g
        nu_dot = Dnu_c + np.matmul(self.Minv, tau_sum)
            
        # Actuator dynamics
        delta_r_dot = (delta_r_c - delta_r) / self.T_delta
        delta_re_dot = (delta_re_c - delta_re) / self.T_delta
        delta_le_dot = (delta_le_c - delta_le) / self.T_delta      #This sets the fin angle as it aproaches the commanded angle. A larger time constant will make it move slower
        n_dot = (n_c - n) / self.T_n

        # Forward Euler integration [k+1]
        nu += sampleTime * nu_dot
        delta_r += sampleTime * delta_r_dot
        delta_re += sampleTime * delta_re_dot
        delta_le += sampleTime * delta_le_dot
        n += sampleTime * n_dot
        
        u_actual = np.array([ delta_r, delta_re, delta_le, n ], float)

        return nu, u_actual


    def stepInput(self, t):
        """
        u_c = stepInput(t) generates step inputs.
                     
        Returns:
            
            u_control = [ delta_r   rudder angle (rad)
                         delta_re    right elevator angle (rad)
                         delta_le    left elevator angle (rad)
                         n          propeller revolution (rpm) ]
        """
        delta_r =  0 * self.D2R      # rudder angle (rad)
        delta_re = -5 * self.D2R      # right elevator angle (rad)
        delta_le = -5 * self.D2R      # left elevator angle (rad)  #TODO: Need to figure out what this is
        n = 3000                    # propeller revolution (rpm)
        
        if t > 50:
            delta_r = 0
            
        if t > 25:
            delta_re = 0     
            delta_le = 0     

        u_control = np.array([ delta_r, delta_re, delta_le, n], float)

        return u_control
    
    
    def depthHeadingAutopilot(self, eta, nu, sampleTime):
        """
        [delta_r, delta_s, n] = depthHeadingAutopilot(eta,nu,sampleTime) 
        simultaneously control the heading and depth of the AUV using control
        laws of PID type. Propeller rpm is given as a step command.
        
        Returns:
            
            u_control = [ delta_r   rudder angle (rad)
                         delta_re    right elevator angle (rad)
                         delta_le    left elevator angle (rad)
                         n          propeller revolution (rpm) ]
            
        """
        z = eta[2]                  # heave position (depth)
        theta = eta[4]              # pitch angle
        psi = eta[5]                # yaw angle
        q = nu[4]                   # pitch rate
        r = nu[5]                   # yaw rate
        e_psi = psi - self.psi_d    # yaw angle tracking error
        e_r   = r - self.r_d        # yaw rate tracking error
        z_ref = self.ref_z          # heave position (depth) setpoint
        psi_ref = self.ref_psi * self.D2R   # yaw angle setpoint
        
        #######################################################################
        # Propeller command
        #######################################################################
        n = self.ref_n 
        
        #######################################################################            
        # Depth autopilot (succesive loop closure)
        #######################################################################
        # LP filtered desired depth command
        self.z_d  = math.exp( -sampleTime * self.wn_d_z ) * self.z_d \
            + ( 1 - math.exp( -sampleTime * self.wn_d_z) ) * z_ref  
            
        # PI controller    
        theta_d = self.Kp_z * ( (z - self.z_d) + (1/self.T_z) * self.z_int )
        # delta_s = -self.Kp_theta * ssa( theta - theta_d ) - self.Kd_theta * q \
        #     - self.Ki_theta * self.theta_int
        delta_re = 0  #TODO: this is the hard part
        delta_le = 0  #TODO: this is the hard part


        # Euler's integration method (k+1)
        self.z_int     += sampleTime * ( z - self.z_d )
        self.theta_int += sampleTime * ssa( theta - theta_d )

        #######################################################################
        # Heading autopilot (PID controller)
        #######################################################################
        
        wn = self.wn_psi            # PID natural frequency
        zeta = self.zeta_psi        # PID natural relative damping factor
        wn_d = self.wn_d            # reference model natural frequency
        zeta_d = self.zeta_d        # reference model relative damping factor

        m = self.M[5][5]           
        d = 0  
        k = 0

        # PID feedback controller with 3rd-order reference model
        [delta_r, self.e_psi_int, self.psi_d, self.r_d, self.a_d] = \
            PIDpolePlacement( 
                self.e_psi_int, 
                e_psi, e_r, 
                self.psi_d, 
                self.r_d, 
                self.a_d, 
                m, 
                d, 
                k, 
                wn_d, 
                zeta_d, 
                wn, 
                zeta, 
                psi_ref, 
                self.r_max, 
                sampleTime 
                )
                
        # Euler's integration method (k+1)
        self.e_psi_int += sampleTime * ssa( psi - self.psi_d )
        
        
        u_control = np.array([ delta_r, delta_re, delta_le, n], float)

        return u_control

    
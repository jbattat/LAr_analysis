import math # fixme: replace with numpy (for array-based calcs)

#########################################
# Electron transport calulations
#########################################

def mobility(temperature, Efield):
    # Compute electron mobility in LAr
    # 
    # Inputs: 
    #    temperature: [K] LAr temperature
    #    Efield: [V/cm] Drift electric field
    # Outputs:
    #    electron mobility #cm^2/V/s
    #
    # From: https://lar.bnl.gov/properties/trans.html
    Efield2 = Efield*1e-3 # convert to kV/cm

    a0, a1, a2 = 551.6, 7158.3, 4440.43  # cm^2/V/s
    a3, a4, a5 = 4.29, 43.63, 0.2053
    T0 = 89.0 # Kelvin
    
    num = a0 + a1*Efield2 + a2*Efield2**(1.5) + a3*Efield2**(2.5)
    den = 1 + (a1/a0)*Efield2 + a4*Efield2**2 + a5*Efield2**3

    return (num / den) * (T0/temperature)**(1.5)  # cm^2/V/s

def electron_drift_speed(mobility, Efield):
    # Compute electron drift speed in LAr
    # 
    # Inputs: 
    #    mobility: [cm^2/V/s] electron mobility 
    #    Efield: [V/cm] Drift electric field
    #
    # Outputs:
    #    electron drift speed in cm/us
    #
    # From: https://lar.bnl.gov/properties/trans.html
    return mobility*Efield*1e-6  # cm/us

################################
### JB STOPPED HERE ###
################################
def attachment_rate(Efield):
    # Compute electron attachment rate
    # 
    # Inputs: 
    #    Efield: [V/cm] Drift electric field
    # 
    # Outputs:
    #    electron attachment rate [1/s]
    #
    # From: https://lar.bnl.gov/properties/trans.html

    Efield2 = Efield*1e-3  # Convert to kV/cm
    p = 11
    a1, a2, a3, a4 = 39.4, 1.20062, 0, 0
    b1, b2, b3, b4 = 0.925794, 1.63816, 0, 0

    num = (a1/b1) + a1 * Efield2 + a2 * (Efield2**2)+a3 * (Efield2**3)+a4 * (Efield2**4)
    den = 1 + b1*Efield2 + b2 * (Efield2**2) + b3 * (Efield2**3) + b4 * (Efield2**4)
    attachment_rate = (10**p) * (num/den)
    
    #numerator = (76.2749 / 1.88083) + 76.2749 * field_voltage + 4.24596 * (field_voltage ** 2)
    #denominator = 1 + 1.88083 * field_voltage + 2.62643 * (field_voltage ** 2) + 0.0632332 * (field_voltage ** 3) - 0.000211009 * (field_voltage ** 4)
    #attachment_rate = (10 ** 11) * (numerator / denominator)
    return attachment_rate #/s



# Attachment rate constant
def calculate_attachment_rate(field):
    field_voltage = field / 1000  # Convert field

    numerator = (76.2749 / 1.88083) + 76.2749 * field_voltage + 4.24596 * (field_voltage ** 2)
    denominator = 1 + 1.88083 * field_voltage + 2.62643 * (field_voltage ** 2) + 0.0632332 * (field_voltage ** 3) - 0.000211009 * (field_voltage ** 4)

    attachment_rate = (10 ** 11) * (numerator / denominator)

    return attachment_rate #/s

# Electron lifetime
def calculate_lifetime(attachment_rate, o2):
    denominator = attachment_rate * o2 * 0.000000001  # Convert O2 to ppm
    lifetime = (1 / denominator) * 1000000  # Convert to µs
    return lifetime #µs

# Drift distance
def calculate_drift_distance(drift_velocity, lifetime):
    drift_distance = drift_velocity * lifetime
    return drift_distance #cm

    # Hardcoded values
temperature = 87  # K
field = 160  # V/cm
o2 = 1  # ppb

emobility = mobility(temperature, field)  # cm^2/V
drift_speed = electron_drift_speed(emobility, field)  # cm/us

attachment_rate = calculate_attachment_rate(field)  # /s
lifetime = calculate_lifetime(attachment_rate, o2)  # us
drift_distance = calculate_drift_distance(drift_speed, lifetime)  # cm


#/////
#Distances caluations
#/////

# Cathode to cathode grid
def calculate_E1(cathode, cathode_to_grid):
  E1 = -1 * cathode / cathode_to_grid
  return E1 #V/cm

# Cathode grid to anode grid
def calculate_E2(anode_grid, grid_to_grid):
  E2 = anode_grid / grid_to_grid
  return E2 #V/cm

# Anode to anode grid
def calculate_E3(anode, anode_grid, anode_to_grid):
  numerator = anode - anode_grid
  E3 = numerator / anode_to_grid
  return E3 #V/cm

#E2/E1
def calculate_E2E1(E2, E1):
  E2E1 = E2 / E1
  return E2E1

#E3/E2
def calculate_E3E2(E2, E3):
  E3E2 = E3 / E2
  return E3E2

#E1, cathode to cathode grid GIVEN PREAMP
def calculate_E1P(E2P, min_field_ratio):
  E1P = E2P / min_field_ratio
  return E1P #V/cm

#E3, anode to anode grid GIVEN PREAMP
def calculate_E3P(E2P, min_field_ratio):
  E3P = E2P * min_field_ratio
  return E3P #V/cm

#Cathode voltage given E1P/preamp
def calculate_cathode(E1P, cathode_to_grid):
  cathode = -1 * E1P * cathode_to_grid
  return cathode #V

#Anode grid voltage given E2P/preamp
def calculate_anode_grid(E2P, grid_to_grid):
  anode_grid = E2P * grid_to_grid
  return anode_grid #V

#Anode voltage given preamp
def calculate_anode(E3P, anode_to_grid, anode_grid):
  anode = (E3P * anode_to_grid) + anode_grid
  return anode #V

# Hardcoded values
cathode = -300 #V
anode_grid = 956 #V
anode = 1355 #V
cathode_to_grid = 1.65 #cm
grid_to_grid = 5.98 #cm
anode_to_grid = 1.03 #cm
E2P = 150 #Cathode grid to anode grid, V/cm)
min_field_ratio = 2.42

# Calculate results
E1 = calculate_E1(cathode, cathode_to_grid) #V/cm
E2 = calculate_E2(anode_grid, grid_to_grid) #V/cm
E3 = calculate_E3(anode, anode_grid, anode_to_grid) #V/cm
E2E1 = calculate_E2E1(E2, E1) #ratio, V/cm
E3E2 = calculate_E3E2(E2, E3) #ratio, V/cm
E1P = calculate_E1P(E2P, min_field_ratio) #V/cm
E3P = calculate_E3P(E2P, min_field_ratio) #V/cm
cathode = calculate_cathode(E1P, cathode_to_grid) #V
anode_grid = calculate_anode_grid(E2P, grid_to_grid) #V
anode = calculate_anode(E3P, anode_to_grid, anode_grid) #V

#////
#Electron Lifetime Measurement Calculator
#////

# Electron lifetime
def calculate_lifetime(anode_amp, cathode_amp, feedback, decay_time, t1, t2, t3):
    #calculations
    anode_charge = anode_amp * feedback * t3/decay_time / (1 - math.exp(-t3/decay_time))
    cathode_charge = (cathode_amp * feedback * t1) / decay_time / (1 - math.exp(-t1 / decay_time))
    numerator = (t2 + (t1 + t3) / 2)
    denominator = math.log(anode_charge / cathode_charge)
    lifetime = -(numerator / denominator)
    return lifetime

# Attachment rate constant
def calculate_attachment_rate_old(field):
    field_voltage = field / 1000  # Convert field
    numerator = (76.2749 / 1.88083) + 76.2749 * field_voltage + 4.24596 * (field_voltage ** 2)
    denominator = 1 + 1.88083 * field_voltage + 2.62643 * (field_voltage ** 2) + 0.0632332 * (field_voltage ** 3) - 0.000211009 * (field_voltage ** 4)
    attachment_rate = (10 ** 11) * (numerator / denominator)
    return attachment_rate #/s

# O2 concentration
def calculate_o2_concentration(field, anode_amp, cathode_amp, feedback, decay_time, t1, t2, t3):
    attachment_rate = calculate_attachment_rate(field)
    electron_lifetime = calculate_lifetime(anode_amp, cathode_amp, feedback, decay_time, t1, t2, t3)
    o2_concentration = 1 / (attachment_rate * electron_lifetime * 1e-6) * 1e9
    return o2_concentration #ppb

# o2: torr to ppb conversion
def calculate_o2(initial_vacuum_level, initial_o2_percentage, chamber_volume, temperature):
    numerator = ((initial_vacuum_level * initial_o2_percentage * 133.322) * chamber_volume)
    denominator = (8.134 * temperature * 318.91)
    o2 = (numerator / denominator) * 10**9
    return o2 #ppb

# Calculate LAr
def calculate_LAr(pressure):
    LAr = 856.38 / (12.47 - math.log(pressure)) - 273.15
    return LAr #C

# Main block
field = 160
#lifetime inputs
anode_amp = 20.7 #mV
cathode_amp = 43.3 #mV
t1 = 70 #µs
t2 = 72 #µs
t3 = 17 #µs
feedback = 1.4 #pF
decay_time = 140 #µs
#o2 convert inputs
initial_vacuum_level = 2.85e-5 #torr
initial_o2_percentage = 0.21
chamber_volume = 1.25e-2 #m^3
temperature = 300 #K
#LAr inputs
pressure = 24 #psia

#calculate
lifetime = calculate_lifetime(anode_amp, cathode_amp, feedback, decay_time, t1, t2, t3)
attachment_rate = calculate_attachment_rate(field)
o2_concentration = calculate_o2_concentration(field, anode_amp, cathode_amp, feedback, decay_time, t1, t2, t3)
o2 = calculate_o2(initial_vacuum_level, initial_o2_percentage, chamber_volume, temperature)
LAr = calculate_LAr(pressure)

#print to test calculators
print(E1P)

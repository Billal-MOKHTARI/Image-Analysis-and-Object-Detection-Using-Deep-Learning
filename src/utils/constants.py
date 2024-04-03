import re

VALID_UNITS = [
    'mm', 'cm', 'm', 'km', 'in', 'ft', 'mi',  # Length
    'mg', 'g', 'kg', 'oz', 'lb', 'st',  # Weight or Mass
    'mL', 'L', 'gal', 'qt', 'pt', 'fl oz',  # Volume
    'm/s', 'km/h', 'mph', 'knot',  # Speed
    'ha',  # Area
    'L', 'mL',  # Volume (Cubic)
    'Hz', 'kHz', 'MHz', 'GHz',  # Frequency
    'Pa', 'kPa', 'MPa', 'GPa', 'bar', 'atm', 'psi',  # Pressure
    'N', 'kN', 'mN', 'lbf', 'dyne',  # Force
    'J', 'kJ', 'cal', 'kcal', 'BTU',  # Energy
    'W', 'mW', 'kW', 'MW', 'GW', 'TW',  # Power
    'V', 'mV', 'kV', 'MV', 'GV',  # Voltage
    'A', 'mA', 'kA', 'MA', 'GA',  # Current
    'Ω', 'mΩ', 'kΩ', 'MΩ', 'GΩ',  # Resistance
    'F', 'mF', 'μF', 'nF', 'pF',  # Capacitance
    'H', 'mH', 'μH', 'nH', 'pH',  # Inductance
    'C', 'mC', 'μC', 'nC', 'pC',  # Electric Charge
    'T', 'mT', 'μT', 'nT', 'pT',  # Magnetic Flux Density
    'mol', 'mmol', 'kmol',  # Amount of Substance
    'rad', 'deg', 'rev',  # Angle
    's', 'ms', 'μs', 'ns', 'ps', 'fs', 'as',  # Time
    '°C', '°F', 'K',  # Temperature
    'dB', 'dBm',  # Decibel
    'g-force',  # Acceleration
    'lx', 'lm', 'cd',  # Light
    'rpm',  # Rotational Speed
    'cc', 'kVAr', 'MVar', 'GVar', 'VA', 'kVA', 'MVA', 'GVA', 'VAR', 'mVAR', 'uVAR',  # Miscellaneous
    # Imperial units
    'thou', 'line', 'inch', 'foot', 'yard', 'chain', 'furlong', 'mile', 'league',
    'acre', 'rood', 'square', 'fathom', 'link', 'rod', 'perch', 'pole', 'furlong',
    'fluid ounce', 'gill', 'pint', 'quart', 'gallon', 'minim', 'drachm', 'scruple',
    'ounce', 'pound', 'stone', 'hundredweight', 'ton',
    # Ancient units
    'cubit', 'digit', 'palm', 'hand', 'span', 'foot', 'step', 'mile', 'league',
    'talent', 'mina', 'shekel', 'gerah',
    'DX', 'DmkIII', 'DmkIV', 'DmkV', 'DmkVI', 'DmkVII', 'DmkVIII', 'DmkIX', 'DmkX', 'd',
    'b', 'kb', 'Mb', 'Gb', 'Tb', 'Pb', 'Eb', 'Zb', 'Yb',  # Data
]

# Constructing the regex pattern
valid_units_pattern = '|'.join(map(re.escape, VALID_UNITS))
unit_regex_pattern = rf'\(?([0-9]*({valid_units_pattern})[0-9]*/)*[0-9]*({valid_units_pattern})[0-9]*\)?'
unit_regex_pattern = re.compile(unit_regex_pattern, re.IGNORECASE)
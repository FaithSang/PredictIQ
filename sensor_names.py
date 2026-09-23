"""
NASA C-MAPSS sensor ID -> human-readable name/description.
Source: NASA Prognostics Center of Excellence, C-MAPSS dataset documentation.
Used to translate raw sensor columns (s1..s21) into plain-English labels for the
SHAP explanations and the Day 4 Claude API advisory layer.
"""

SENSOR_INFO = {
    "set1": {"name": "Altitude setting",              "symbol": "alt"},
    "set2": {"name": "Mach number setting",           "symbol": "mach"},
    "set3": {"name": "Throttle resolver angle",       "symbol": "TRA"},
    "s1":  {"name": "Fan inlet temperature",        "symbol": "T2"},
    "s2":  {"name": "LPC outlet temperature",        "symbol": "T24"},
    "s3":  {"name": "HPC outlet temperature",        "symbol": "T30"},
    "s4":  {"name": "LPT outlet temperature",        "symbol": "T50"},
    "s5":  {"name": "Fan inlet pressure",            "symbol": "P2"},
    "s6":  {"name": "Bypass-duct pressure",          "symbol": "P15"},
    "s7":  {"name": "HPC outlet pressure",           "symbol": "P30"},
    "s8":  {"name": "Physical fan speed",            "symbol": "Nf"},
    "s9":  {"name": "Physical core speed",           "symbol": "Nc"},
    "s10": {"name": "Engine pressure ratio",         "symbol": "epr"},
    "s11": {"name": "HPC outlet static pressure",    "symbol": "Ps30"},
    "s12": {"name": "Fuel flow / Ps30 ratio",        "symbol": "phi"},
    "s13": {"name": "Corrected fan speed",           "symbol": "NRf"},
    "s14": {"name": "Corrected core speed",          "symbol": "NRc"},
    "s15": {"name": "Bypass ratio",                  "symbol": "BPR"},
    "s16": {"name": "Burner fuel-air ratio",         "symbol": "farB"},
    "s17": {"name": "Bleed enthalpy",                "symbol": "htBleed"},
    "s18": {"name": "Demanded fan speed",            "symbol": "Nf_dmd"},
    "s19": {"name": "Demanded corrected fan speed",  "symbol": "PCNfR_dmd"},
    "s20": {"name": "HPT coolant bleed",             "symbol": "W31"},
    "s21": {"name": "LPT coolant bleed",             "symbol": "W32"},
}


def describe(sensor_col: str) -> str:
    """e.g. 's13' -> 'Corrected fan speed (NRf)'"""
    info = SENSOR_INFO.get(sensor_col)
    if not info:
        return sensor_col
    return f"{info['name']} ({info['symbol']})"

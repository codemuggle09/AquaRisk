import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def fuzzy_classify_fluoride(fluoride_value):
    # ---------- UNIVERSE ----------
    fluoride = ctrl.Antecedent(np.arange(0, 4.1, 0.01), 'fluoride')
    risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

    # ---------- MEMBERSHIP FUNCTIONS ----------
    # Smooth, continuous, overlapping — NO GAPS
    fluoride['very_low']  = fuzz.trimf(fluoride.universe, [0.0, 0.0, 0.5])
    fluoride['low']       = fuzz.trimf(fluoride.universe, [0.3, 0.7, 1.1])
    fluoride['normal']    = fuzz.trimf(fluoride.universe, [0.9, 1.5, 2.1])
    fluoride['high']      = fuzz.trimf(fluoride.universe, [1.8, 2.4, 3.0])
    fluoride['very_high'] = fuzz.trimf(fluoride.universe, [2.8, 3.5, 4.0])

    # Risk membership
    risk['low']      = fuzz.trimf(risk.universe, [0, 0, 40])
    risk['medium']   = fuzz.trimf(risk.universe, [20, 50, 80])
    risk['high']     = fuzz.trimf(risk.universe, [60, 100, 100])

    # ---------- RULES ----------
    rule1 = ctrl.Rule(fluoride['very_low'],  risk['low'])
    rule2 = ctrl.Rule(fluoride['low'],       risk['medium'])
    rule3 = ctrl.Rule(fluoride['normal'],    risk['low'])
    rule4 = ctrl.Rule(fluoride['high'],      risk['medium'])
    rule5 = ctrl.Rule(fluoride['very_high'], risk['high'])

    # ---------- CONTROL SYSTEM ----------
    risk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
    risk_sim = ctrl.ControlSystemSimulation(risk_ctrl)

    # ---------- INPUT ----------
    risk_sim.input['fluoride'] = float(fluoride_value)

    # ---------- COMPUTE SAFELY ----------
    try:
        risk_sim.compute()
        score = risk_sim.output['risk']
    except Exception as e:
        return ("Computation Error", -1)

    # ---------- LABEL OUTPUT ----------
    if score < 33:
        label = "Low Risk"
    elif score < 66:
        label = "Medium Risk"
    else:
        label = "High Risk"

    return label, round(score, 2)

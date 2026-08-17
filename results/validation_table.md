| gate | quantity | value | units | target | pass | note |
|---|---|---|---|---|---|---|
| healthy baseline | ejection_fraction | 0.6584 | fraction | 0.55 to 0.7 | True |  |
| healthy baseline | edv_ml | 112.9699 | mL | 110 to 130 | True |  |
| healthy baseline | stroke_volume_ml | 74.3824 | mL | 65 to 75 | True |  |
| healthy baseline | peak_lv_pressure_mmhg | 110.7581 | mmHg | 110 to 130 | True |  |
| healthy baseline | end_diastolic_pressure_mmhg | 9.5774 | mmHg | 5 to 12 | True |  |
| healthy baseline | mean_arterial_pressure_mmhg | 94.9093 | mmHg | 85 to 95 | True |  |
| healthy baseline | cardiac_output_l_per_min | 4.8456 | L/min | 4.5 to 5.5 | True |  |
| healthy baseline | wall_thickness_cm | 0.9245 | cm | 0.6 to 1.1 | True |  |
| healthy baseline | not obstructive | 1.0 | boolean | true | True | peak gradient 7.1 mmHg |
| healthy baseline | steady state reached | 1.0 | boolean | true | True | 6 beats |
| ejection realism | peak_lvot_velocity_m_per_s | 1.3324 | m/s | 0.7 to 1.4 | True |  |
| ejection realism | ejection_duration_ms | 260.7692 | ms | 250 to 340 | True |  |
| ejection realism | peak_aortic_flow_ml_per_s | 599.593 | mL/s | 350 to 600 | True |  |
| Frank-Starling | EDV rises with preload | 1.0 | boolean | true | True | EDV 100.5, 108.9, 116.9, 124.5 mL |
| Frank-Starling | SV rises with preload | 1.0 | boolean | true | True | SV 67.9, 72.4, 76.3, 79.6 mL |
| afterload | SV falls with afterload | 1.0 | boolean | true | True | SV 79.7, 74.4, 69.7, 65.6 mL |
| afterload | ESV rises with afterload | 1.0 | boolean | true | True | ESV 36.7, 38.6, 40.2, 41.5 mL |
| loop shape | loop closes | 1.0 | boolean | true | True | gap 0.107 mL over 74.5 mL |
| loop shape | counter-clockwise | 1.0 | boolean | true | True | stroke work 6861 mmHg*mL |
| loop shape | isovolumic phases present | 1.0 | boolean | true | True | 36 contraction, 107 relaxation steps |
| diastolic | passive relation monotone | 1.0 | boolean | true | True | at fixed volume |
| diastolic | passive relation convex | 1.0 | boolean | true | True | at fixed volume |
| diastolic | stiffer tissue raises filling pressure | 1.0 | boolean | true | True | EDP 8.5, 9.6, 11.0, 12.5, 13.9 mmHg |
| HCM phenotype emerges | supranormal ejection fraction | 1.0 | boolean | true | True | 0.747 vs healthy 0.658 |
| HCM phenotype emerges | reduced stroke volume | 1.0 | boolean | true | True | 60.1 vs 74.4 mL |
| HCM phenotype emerges | elevated filling pressure | 1.0 | boolean | true | True | 13.3 vs 9.6 mmHg |
| HCM phenotype emerges | elevated E/e' surrogate | 1.0 | boolean | true | True | 10.8 vs 7.3 |
| HCM phenotype emerges | reduced strain despite preserved EF | 1.0 | boolean | true | True | 0.154 vs 0.209 |
| HCM phenotype emerges | elevated energy cost per unit work | 1.0 | boolean | true | True | 2.05x healthy |
| HCM phenotype emerges | hypertrophic on imaging | 1.0 | boolean | true | True | 1.59 cm |
| HCM phenotype emerges | disease needs material, not only geometry | 1.0 | boolean | true | True | thick wall with healthy tissue: EDP 9.0 mmHg, EF 0.758 |
| drug direction | dose lowers ejection fraction | 1.0 | boolean | true | True | EF 0.747, 0.730, 0.715, 0.694, 0.679 |
| drug direction | dose lowers outflow gradient | 1.0 | boolean | true | True | gradient 76, 59, 48, 37, 30 mmHg |
| exposure-response (PREDICTION, not fitted) | ejection-fraction change at the mid dose | -5.34 | percentage points | -20 to -0.5; trial reference -4.8 (SEQUOIA-HCM, aficamten) | True | no model parameter was calibrated against a published dose-response curve |
| exposure-response (PREDICTION, not fitted) | outflow-gradient change at the mid dose | -39.4 | mmHg | < -10; trial reference about -35 (SEQUOIA-HCM) | True | reported for comparison, not asserted quantitatively |

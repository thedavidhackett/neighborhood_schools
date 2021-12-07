import pandas as pd

data = pd.read_csv("district_charter_enrollment.csv")

stl_data = data[data['COUNTY_DISTRICT_CODE'] == 115115]

stl_data = stl_data[stl_data['YEAR'] >= 2000]
stl_data = stl_data[["YEAR", "ENROLLMENT_GRADES_K_12"]]

stl_data.to_csv("stl_public_school_enrollment.csv", index=False)

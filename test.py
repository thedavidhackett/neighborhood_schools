from time_series import TimeSeries

stl_data = TimeSeries("stl_public_school_enrollment.csv", True, (0,1), data_label="SLPS Enrollment")


# import numpy as np
# values = stl_data._data[:, [0]]
# ones = np.ones((len(values), 1))
# A = np.concatenate([ones, values], axis=1)
# print(A[0])
print(stl_data.linear_regression(7, True, False))


# from prediction_data import PredictionData


# stl_schools = PredictionData("stl_neighborhood_schools.csv", [i for i in range(2,19)])
# stl_schools.standardize_multiple(["ENROLLMENT_CHANGE", "ENROLLMENT_CHANGE"])

# print(stl_schools.linear_regression("ENROLLMENT_CHANGE", "WARD_CHILD_POP_CHANGE_PCT", False))

# from time_series import TimeSeries

# stl_data = TimeSeries("stl_public_school_enrollment.csv", True, (0,1), data_label="SLPS Enrollment")


# print(stl_data.predict_point_n(7, True, 2022))


from prediction_data import PredictionData


stl_schools = PredictionData("stl_neighborhood_schools.csv", [i for i in range(2,19)])
stl_schools.standardize_multiple(["ENROLLMENT_CHANGE", "ENROLLMENT_CHANGE"])
stl_schools.linear_regression("ENROLLMENT_CHANGE", "WARD_CHILD_POP_CHANGE_PCT")

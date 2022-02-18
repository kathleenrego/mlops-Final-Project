import pandas as pd
import scipy.stats

# Non Deterministic Test
def test_kolmogorov_smirnov(data, ks_alpha):

    sample1, sample2 = data
    
    numerical_columns = [
        "age",
        "anaemia",
        "creatinine_phosphokinase",
        "diabetes",
        "ejection_fraction",
        "high_blood_pressure",
        "platelets",
        "serum_creatinine",
        "serum_sodium",
        "sex",
        "smoking",
        "time",
        ]
    
    # Bonferroni correction for multiple hypothesis testing
    alpha_prime = 1 - (1 - ks_alpha)**(1 / len(numerical_columns))

    for col in numerical_columns:

        # two-sided: The null hypothesis is that the two distributions are identical
        # the alternative is that they are not identical.
        ts, p_value = scipy.stats.ks_2samp(
            sample1[col],
            sample2[col],
            alternative='two-sided'
        )

        # NOTE: as always, the p-value should be interpreted as the probability of
        # obtaining a test statistic (TS) equal or more extreme that the one we got
        # by chance, when the null hypothesis is true. If this probability is not
        # large enough, this dataset should be looked at carefully, hence we fail
        assert p_value > alpha_prime
        
# Determinstic Test
def test_column_presence_and_type(data):
    
    # Disregard the reference dataset
    _, df = data

    required_columns = {
        "age": pd.api.types.is_float_dtype,
        "anaemia": pd.api.types.is_int64_dtype,
        "creatinine_phosphokinase": pd.api.types.is_int64_dtype,
        "diabetes": pd.api.types.is_int64_dtype,
        "ejection_fraction": pd.api.types.is_int64_dtype,
        "high_blood_pressure": pd.api.types.is_int64_dtype,
        "platelets": pd.api.types.is_float_dtype,
        "serum_creatinine": pd.api.types.is_float_dtype,
        "serum_sodium": pd.api.types.is_int64_dtype,
        "sex": pd.api.types.is_int64_dtype,
        "smoking": pd.api.types.is_int64_dtype,
        "time": pd.api.types.is_int64_dtype,
        "DEATH_EVENT": pd.api.types.is_int64_dtype
    }

    # Check column presence
    assert set(df.columns.values).issuperset(set(required_columns.keys()))

    for col_name, format_verification_funct in required_columns.items():

        assert format_verification_funct(df[col_name]), f"Column {col_name} failed test {format_verification_funct}"

# Deterministic Test
def test_class_names(data):
    
    # Disregard the reference dataset
    _, df = data

    # Check that only the known classes are present
    known_classes = [
        0,
        1
    ]

    assert df["DEATH_EVENT"].isin(known_classes).all()

# Deterministic Test
def test_column_ranges(data):
    
    # Disregard the reference dataset
    _, df = data

    ranges = {
        "age": (40.0, 95.0),
        "creatinine_phosphokinase": (23.0, 7861.0),
        "ejection_fraction": (14.0, 80.0),
        "platelets": (25100.0, 850000.0),
        "serum_creatinine": (0.5, 9.4),
        "serum_sodium": (113.0, 148.0),
        "time": (4.0, 285.0)
    }

    for col_name, (minimum, maximum) in ranges.items():

        assert df[col_name].dropna().between(minimum, maximum).all(), (
            f"Column {col_name} failed the test. Should be between {minimum} and {maximum}, "
            f"instead min={df[col_name].min()} and max={df[col_name].max()}"
        )

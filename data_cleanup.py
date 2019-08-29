import pandas as pd
from numpy import nan

# File we are loading
FILE = "Tiago-Martino_Collab.csv"
# File we'll be saving
TARGET_FILE = "dataset_cleaned.csv"
# Set the columns to be used as they are
COLS_USE = ["Age", "Mat_Indiff", "Mat_Abus", "Mat_Over", "ACES",
            "Avoi", "Anx", "PHQ", "Defeat", "Entrp",
            "Burd", "Thw_Belon", "Strategy", "Non_Accept", "Impulse",
            "Goals", "Aware", "Clarity", "SPS", "SI_ever"]
# Set columns for which we need to use dummies (turn into binary)
# note that it's not enough to just add them here, you also need to manually
# turn them into dummies below.
COLS_USE_DUMMIES = ["SexualO", "Psych_D", "Psych_M"]

# load the data
data = pd.read_csv(FILE)
# set the index as the Part_ID
data.set_index("Part_ID", inplace=True)
# we don't need the "Part_Code" column
data.drop(["Part_Code"], 1, inplace=True)
# for some reason the CSV has some unnamed empty cols, we remove them
unn = [colname for colname in data.columns if colname[:7] == "Unnamed"]
data.drop(unn, 1, inplace=True)
# turn the "Unknowns" and "Undisclosed" into NAN
data.replace("Unknown", nan, inplace=True)
data.replace("Undisclosed", nan, inplace=True)
# keep only the columns we are interested in using
reduced_df = data[COLS_USE_DUMMIES + COLS_USE]
print("Size prior to dropping missing data:", len(reduced_df))
# dropping rows with NaN
non_null = ~reduced_df.isnull().any(axis=1)
reduced_df = reduced_df[non_null]
print("Size after dropping missing data:", len(reduced_df))
# Additional columns, turned into binary
reduced_df = reduced_df.assign(
    Heterosexual=(reduced_df["SexualO"] == "Heterosexual").astype(int))
reduced_df = reduced_df.assign(
    Psych_D_Yes=(reduced_df["Psych_D"] == "Yes").astype(int))
reduced_df = reduced_df.assign(
    Psych_M_Yes=(reduced_df["Psych_M"] == "Yes").astype(int))
# remove the columns that we no longer need because we added the dummies
reduced_df.drop(COLS_USE_DUMMIES, 1, inplace=True)


# we will translate shortened variable names to long versions
TRANSL_D = {"Age": "Age",
            "Mat_Indiff": "Maternal indifference",
            "Mat_Abus": "Maternal abuse",
            "Mat_Over": "Maternal overcontrol",
            "ACES": "Adverse childhood",
            "Avoi": "Attachment avoidance",
            "Anx": "Attachment anxiety",
            "PHQ": "Depressive symptoms",
            "Defeat": "Defeat",
            "Entrp": "Entrapment",
            "Burd": "Burdensomeness",
            "Thw_Belon": "Thwarted belongingness",
            "Strategy": "Strategy",
            "Non_Accept": "Non-acceptance",
            "Impulse": "Impulsivity",
            "Goals": "Establishing goals",
            "Aware": "Establishing awareness",
            "Clarity": "Establishing clarity",
            "SPS": "Current suicidal ideation",
            "SI_ever": "Lifetime suicidal ideation",
            "Heterosexual": "Heterosexual",
            "Psych_D_Yes": "Psychiatric diagnosis",
            "Psych_M_Yes": "Antidepr or anxiolytic"}

reduced_df.columns = [TRANSL_D[col] for col in reduced_df.columns]

reduced_df.to_csv(TARGET_FILE)

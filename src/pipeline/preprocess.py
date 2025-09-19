import pandas as pd

def filter_sites_by_threshold(df: pd.DataFrame, site_col: str, threshold: int) -> pd.DataFrame:
    """
    Keep only rows from sites with number of observations > threshold.

    Args:
        df (pd.DataFrame): Input dataframe.
        site_col (str): Column name with site IDs.
        threshold (int): Minimum number of observations required for a site to be kept.

    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    site_counts = df[site_col].value_counts()
    keep_sites = site_counts[site_counts > threshold].index
    return df[df[site_col].isin(keep_sites)]


def add_income_group(data: pd.DataFrame, t1: int = 50000, t2: int = 100000) -> pd.DataFrame:

    """
    Add income group column.
    Args:
        pd.DataFrame: Input dataframe.
        t1, t2 - optional values for income group division
    Returns:
        pd.DataFrame: Updated dataframe.
    """

    data['income_group'] = pd.cut(data['IncCombinedMidpoint'],
                                    bins=[-float("inf"), t1, t2, float("inf")],
                                    labels=["low", "medium", "high"]
    )
    return data


def preprocess_features(data: pd.DataFrame, g_score: pd.DataFrame, eventname: str) -> pd.DataFrame:

    features = data.copy()

    # all the values of "Task" feature are "rest", we do not need this column
    features = features.drop(columns=['Task'], axis = 0, inplace=False)

    g_factor = g_score.copy()
    g_factor = g_factor.rename(columns={'subjectkey': 'src_subject_id'})

    result_df = features[features['eventname'] == eventname]
    result_df_fg = pd.merge(g_factor, result_df, on='src_subject_id')

    result_df_fg = result_df_fg.drop(columns=['src_subject_id', 'eventname', 'site_id_l', 'Session','rel_family_id'])
    if eventname == 'baseline_year_1_arm_1':
        result_df_fg = result_df_fg.drop(columns=['site_id_l.2Year','G_lavaan.2Year'])
        result_df_fg = result_df_fg.rename(columns={'site_id_l.baseline': 'site_id_l', 'G_lavaan.baseline': 'g_lavaan'})

    elif eventname == '2_year_follow_up_y_arm_1':
        #result_df_fg['rel_family_id'] = df_baseline['Subject'].map(df_baseline.set_index('Subject')['rel_family_id'])
        result_df_fg = result_df_fg.drop(columns=['site_id_l.baseline','G_lavaan.baseline'])
        result_df_fg = result_df_fg.rename(columns={'site_id_l.2Year': 'site_id_l', 'G_lavaan.2Year': 'g_lavaan'})
    # drop subjects with missing values, no replacement
    result_df_fg = result_df_fg.dropna()
    # keep only subjects with 4 or more minutes of noncensored data
    result_df_fg = result_df_fg[result_df_fg['confounds_nocensor'] >= 240]
    if eventname == 'baseline_year_1_arm_1':
        threshold = 75
    elif eventname == '2_year_follow_up_y_arm_1':
        threshold = 50
    result_df_fg = filter_sites_by_threshold(result_df_fg, site_col="site_id_l", threshold=threshold)
    result_df_fg = add_income_group(result_df_fg)

    return result_df_fg


def print_data_summary(data: pd.DataFrame) -> None:
    print(f"Number of subjects: {len(data)}")
    print(f"Interview age mean: {data['interview_age'].mean()/12.:2f}")
    print(f"Interview age std: {data['interview_age'].std()/12:2f}")
    counts_sex = data['demo_sex_v2'].value_counts()
    for sex, n in counts_sex.items():
        print(f"{sex}: {n}")
        print(f"Percentage {sex}: {n/data.shape[0]*100:.2f}%")
    counts_race = data['race.4level'].value_counts()
    for race, n in counts_race.items():
        print(f"{race}: {n}")
        print(f"Percentage {race}: {n/data.shape[0]*100:.2f}%")
    counts_hispanic = data['hisp'].value_counts()
    for hispanic, n in counts_hispanic.items():
        print(f"{hispanic}: {n}")
        print(f"Percentage {hispanic}: {n/data.shape[0]*100:.2f}%")
    counts_income = data['income_group'].value_counts()
    for income, n in counts_income.items():
        print(f"{income}: {n}")
        print(f"Percentage {income}: {n/data.shape[0]*100:.2f}%")

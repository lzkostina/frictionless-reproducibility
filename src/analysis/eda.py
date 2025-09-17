# Function to return a list of columns with missing values in a DataFrame
def list_columns_with_missing_values(df):
    return df.columns[df.isnull().any()].tolist()  # List of columns with any missing values


# Function to count the number of columns with missing values in a DataFrame
def count_columns_with_missing_values(df):
    return (df.isnull().sum() > 0).sum()  # Count columns with any missing values


# Function to calculate the percentage of missing values for each column in a DataFrame
def percentage_missing_values(df, columns):
    missing_percent = df[columns].isnull().mean() * 100  # Percentage of missing values for specified columns
    return missing_percent


# Function to divide features into numerical and categorical
def divide_features(df):
    numerical_features = df.select_dtypes(include=['number']).columns.tolist()  # Select numerical columns
    assert isinstance(df.select_dtypes(include=['object', 'category']).columns, object)
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()  # Select categorical columns (strings or categories)

    return numerical_features, categorical_features

# Function to count unique values for categorical features
def categorical_summary_stats(df, categorical_features):
    return df[categorical_features].nunique()
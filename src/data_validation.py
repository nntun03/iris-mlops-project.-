import pandas as pd
from sklearn.datasets import load_iris

def get_iris_data():
    """Loads and minimally prepares Iris data. In a real scenario, this would read from a DB."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df

def validate_data(df):
    """Validates the data schema and basic assumptions."""
    expected_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'target']
    assert list(df.columns) == expected_columns, "Data columns do not match expected schema."
    assert df.isnull().sum().sum() == 0, "Data contains null values."
    assert df['target'].nunique() == 3, "Target column must have exactly 3 classes."
    print("✅ Data validation passed!")
    return True

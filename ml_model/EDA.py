# EDA.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs("outputs", exist_ok=True)

def analyze_dataframe(df):
    """
    Performs basic analysis on a Pandas DataFrame.
    Args:
        df: The Pandas DataFrame to analyze.
    Returns:
        None. Prints the DataFrame's information, descriptive statistics, 
        missing values, and duplication check.
    """
    print("DataFrame Information:")
    df.info()

    print("\nDescriptive Statistics:")
    print(df.describe())

    print("\nMissing Values:")
    print(df.isna().sum())

    print("\nDuplicate Check:")
    print("Any duplicates: ", df.duplicated().any())

def plot_outliers_boxplots(dataframe, columns, figsize=(16, 20)):
    """
    Plots boxplots for detecting outliers in specified numerical columns.
    """
    plt.figure(figsize=figsize)
    for i, col in enumerate(columns, 1):
        plt.subplot((len(columns) + 1) // 2, 2, i)
        sns.boxplot(x=dataframe[col])
        plt.title(f'Boxplot of {col}')
        plt.tight_layout()
    plt.savefig("outputs/boxplots_outliers.png")    
    plt.show()

def plot_categorical_distributions(df, categorical_cols):
    """
    Plots countplots for each categorical feature.
    """
    plt.figure(figsize=(18, 30))
    for i, col in enumerate(categorical_cols, 1):
        plt.subplot(5, 2, i)
        sns.countplot(y=col, data=df, order=df[col].value_counts().index, palette='Set3')
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
    plt.savefig("outputs/categorical_distributions.png")
    plt.show()

def plot_categorical_vs_target(df, categorical_cols, target_col='y'):
    """
    Plots categorical features against the target variable.
    """
    plt.figure(figsize=(20, 40))
    for i, col in enumerate(categorical_cols, 1):
        plt.subplot(5, 2, i)
        sns.countplot(data=df, x=col, hue=target_col, order=df[col].value_counts().index, palette='Set2')
        plt.title(f'{col.capitalize()} vs Target ({target_col})')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='Subscribed')
        plt.tight_layout()
    plt.savefig("outputs/categorical_vs_target.png")
    plt.show()

def plot_categorical_by_target(data, target_col, categorical_columns, batch_size=5):
    """
    Plots countplots of categorical columns separately for each target class.
    """
    data_yes = data[data[target_col] == 'yes']
    data_no = data[data[target_col] == 'no']

    for start in range(0, len(categorical_columns), batch_size):
        batch = categorical_columns[start:start + batch_size]

        plt.figure(figsize=(18, 5 * len(batch)))
        for i, col in enumerate(batch, 1):
            plt.subplot(len(batch), 2, i)
            sns.countplot(y=col, data=data_yes, order=data_yes[col].value_counts().index, palette='Greens_r')
            plt.title(f'{col} (Target = YES)')
            plt.tight_layout()
        plt.savefig(f"outputs/cat_by_target_yes_batch_{start}.png")
        plt.show()

        plt.figure(figsize=(18, 5 * len(batch)))
        for i, col in enumerate(batch, 1):
            plt.subplot(len(batch), 2, i)
            sns.countplot(y=col, data=data_no, order=data_no[col].value_counts().index, palette='Reds_r')
            plt.title(f'{col} (Target = NO)')
            plt.tight_layout()
        plt.savefig(f"outputs/cat_by_target_no_batch_{start}.png")
        plt.show()

def plot_pie_charts_by_target(data, target_col, categorical_columns, batch_size=6):
    """
    Plots pie charts for categorical columns separately for each target class.
    """
    data_yes = data[data[target_col] == 'yes']
    data_no = data[data[target_col] == 'no']

    for start in range(0, len(categorical_columns), batch_size):
        batch = categorical_columns[start:start + batch_size]

        plt.figure(figsize=(18, 5 * len(batch)))
        for i, col in enumerate(batch, 1):
            plt.subplot((len(batch) + 1) // 2, 2, i)
            data_yes[col].value_counts().plot.pie(
                autopct='%1.1f%%', startangle=90, counterclock=False,
                colors=sns.color_palette("Greens", len(data_yes[col].unique()))
            )
            plt.title(f'Target = YES: {col}')
            plt.ylabel('')
            plt.tight_layout()
        plt.savefig(f"outputs/pie_yes_batch_{start}.png")
        plt.show()

        plt.figure(figsize=(18, 5 * len(batch)))
        for i, col in enumerate(batch, 1):
            plt.subplot((len(batch) + 1) // 2, 2, i)
            data_no[col].value_counts().plot.pie(
                autopct='%1.1f%%', startangle=90, counterclock=False,
                colors=sns.color_palette("Reds", len(data_no[col].unique()))
            )
            plt.title(f'Target = NO: {col}')
            plt.ylabel('')
            plt.tight_layout()
        plt.savefig(f"outputs/pie_no_batch_{start}.png")

        plt.show()

def perform_eda(df_path='bank-additional-full.csv'):
    """
    Main function to perform all EDA tasks.
    """
    df = pd.read_csv(df_path, sep=';')
    
    # Basic analysis
    analyze_dataframe(df)
    
    # Define columns
    numerical_cols = [
        'age', 'campaign', 'pdays', 'previous',
        'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
        'euribor3m', 'nr.employed'
    ]
    
    categorical_cols = [
        'job', 'marital', 'education', 'default', 'housing', 'loan',
        'contact', 'month', 'day_of_week', 'poutcome'
    ]
    
    # Numerical EDA
    plot_outliers_boxplots(df, numerical_cols)
    
    # Age binning
    age_bins = [0, 25, 35, 45, 55, 65, 100]
    age_labels = ['<25', '25-35', '35-45', '45-55', '55-65', '65+']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
    print(df['age_group'].value_counts().sort_index())
    
    # Categorical EDA
    plot_categorical_distributions(df, categorical_cols)
    plot_categorical_vs_target(df, categorical_cols)
    plot_categorical_by_target(df, 'y', categorical_cols)
    plot_pie_charts_by_target(df, 'y', categorical_cols)
    
    return df

if __name__ == "__main__":
    perform_eda()
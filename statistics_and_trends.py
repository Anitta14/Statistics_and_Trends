import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss
import seaborn as sns


def plot_relational_plot(df):
    "Creates a scatter plot of vote_count vs. vote_average."

    fig, ax = plt.subplots(figsize=(10, 6))

    if 'vote_count' in df.columns and 'vote_average' in df.columns:
        sns.scatterplot(
            x=df['vote_count'], y=df['vote_average'], alpha=0.5, ax=ax
        )
        ax.set_xlabel("Vote Count", fontsize=12)
        ax.set_ylabel("Vote Average", fontsize=12)
        ax.set_title("Scatter Plot: Vote Count vs. Vote Average",
                     fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('relational_plot.png')
    else:
        print("Error: Required columns for relational plot not found.")

    return


def plot_categorical_plot(df):
    "Creates bar plot showing average vote_average for each original_language."
    fig, ax = plt.subplots(figsize=(10, 6))

    if 'original_language' in df.columns and 'vote_average' in df.columns:
        lang_avg = (
            df.groupby('original_language')['vote_average']
            .mean()
            .sort_values()
        )
        lang_avg.plot(kind='bar', ax=ax)
        ax.set_xlabel("Original Language", fontsize=12)
        ax.set_ylabel("Average Vote", fontsize=12)
        ax.set_title("Average Vote by Original Language",
                     fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('categorical_plot.png')
    else:
        print("Error: Required columns for categorical plot not found.")

    return


def plot_statistical_plot(df):
    """Creates a box plot for vote_average to analyze its distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if 'vote_average' in df.columns:
        sns.boxplot(x=df['vote_average'], ax=ax)
        ax.set_xlabel("Vote Average", fontsize=12)
        ax.set_title("Box Plot: Vote Average Distribution",
                     fontsize=14)
        plt.tight_layout()
        plt.savefig('statistical_plot.png')
    else:
        print("Error: Required column for statistical plot not found.")

    return


def statistical_analysis(df, col: str):
    "Computes the four main statistical moments for the given column."
    if col in df.columns:
        mean = df[col].mean()
        stddev = df[col].std()
        skew = ss.skew(df[col], nan_policy='omit')
        excess_kurtosis = ss.kurtosis(df[col], nan_policy='omit')
        return mean, stddev, skew, excess_kurtosis
    else:
        print(f"Error: Column {col} not found in the dataset.")
        return None, None, None, None


def preprocessing(df):
    "Prepares dataset by showing summary stats and handling missing values."
    # Keep only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    print(numeric_df.describe())
    print(numeric_df.head())
    print(numeric_df.corr())
    df = df.dropna()  # Remove rows with missing values
    return df


def writing(moments, col):
    "Prints the statistical moments and interpretation for column."
    if None not in moments:
        print(f'For the attribute {col}:')
        print(
            f'Mean = {moments[0]:.2f}, '
            f'Standard Deviation = {moments[1]:.2f},'
        )
        print(
            f'Skewness = {moments[2]:.2f}, '
            f'Excess Kurtosis = {moments[3]:.2f}.'
        )
        skew_desc = (
            "right skewed" if moments[2] > 0 else
            ("left skewed" if moments[2] < 0 else "not skewed")
        )
        kurtosis_desc = (
            "leptokurtic" if moments[3] > 0 else
            ("platykurtic" if moments[3] < 0 else "mesokurtic")
        )
        print(f'The data was {skew_desc} and {kurtosis_desc}.')
    return


def main():
    try:
        df = pd.read_csv('data.csv')
        df = preprocessing(df)
        col = 'vote_average'  # Selected column for statistical analysis
        plot_relational_plot(df)
        plot_statistical_plot(df)
        plot_categorical_plot(df)
        moments = statistical_analysis(df, col)
        writing(moments, col)
    except FileNotFoundError:
        print(
            "Error: The file 'data.csv' was not found. Please ensure "
            "it is in the correct directory."
        )
    except Exception as e:
        print(f"Unexpected error: {e}")
    return


if __name__ == '__main__':
    main()

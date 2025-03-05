import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss
import seaborn as sns


def plot_relational_plot(df):
    """
    Creates a scatter plot showing the relationship between vote_count and vote_average.
    """
    fig, ax = plt.subplots()
    sns.scatterplot(x=df['vote_count'], y=df['vote_average'], alpha=0.5, ax=ax)
    ax.set_xlabel("Vote Count")
    ax.set_ylabel("Vote Average")
    ax.set_title("Scatter Plot: Vote Count vs. Vote Average")
    plt.savefig('relational_plot.png')
    return



def plot_categorical_plot(df):
    """
    Creates a bar plot showing the average vote_average for each original_language.
    """
    fig, ax = plt.subplots()
    lang_avg = df.groupby('original_language')['vote_average'].mean().sort_values()
    lang_avg.plot(kind='bar', ax=ax)
    ax.set_xlabel("Original Language")
    ax.set_ylabel("Average Vote")
    ax.set_title("Average Vote by Original Language")
    plt.xticks(rotation=45)
    plt.savefig('categorical_plot.png')
    return



def plot_statistical_plot(df):
    """
    Creates a box plot for vote_average to analyze its distribution.
    """
    fig, ax = plt.subplots()
    sns.boxplot(x=df['vote_average'], ax=ax)
    ax.set_xlabel("Vote Average")
    ax.set_title("Box Plot: Vote Average Distribution")
    plt.savefig('statistical_plot.png')
    return



def statistical_analysis(df, col: str):
    """
    Computes the four main statistical moments for the given column.
    """
    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col], nan_policy='omit')
    excess_kurtosis = ss.kurtosis(df[col], nan_policy='omit')
    return mean, stddev, skew, excess_kurtosis



def preprocessing(df):
    """
    Preprocesses the dataset by displaying summary statistics and handling missing values.
    """
    print(df.describe())
    print(df.head())
    print(df.corr())
    df = df.dropna()  # Remove rows with missing values
    return df



def writing(moments, col):
    """
    Prints the statistical moments and interpretation for the selected column.
    """
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, Standard Deviation = {moments[1]:.2f}, ')
    print(f'Skewness = {moments[2]:.2f}, Excess Kurtosis = {moments[3]:.2f}.')
    skew_desc = "right skewed" if moments[2] > 0 else "left skewed" if moments[2] < 0 else "not skewed"
    kurtosis_desc = "leptokurtic" if moments[3] > 0 else "platykurtic" if moments[3] < 0 else "mesokurtic"
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
        print("Error: The file 'data.csv' was not found. Please ensure it is in the correct directory.")
    except KeyError as e:
        print(f"Error: Column {e} not found in the dataset. Please check the column names.")
    return


if __name__ == '__main__':
    main()

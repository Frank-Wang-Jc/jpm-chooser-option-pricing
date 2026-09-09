import pandas as pd
import numpy as np

from pathlib import Path


DATA_DIR = Path("../Week1/raw_data")
OUTPUT_DIR = Path("./processed_data")


def remove_invalid_leading_rows(df, date_column="Date"):
    """
    Remove consecutive leading rows whose date values cannot be parsed.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    date_column : str
        Name of the date column.

    Returns
    -------
    pandas.DataFrame
        Dataset beginning with the first valid dated observation.
    """
    cleaned_df = df.copy()

    while not cleaned_df.empty:
        first_date = pd.to_datetime(
            cleaned_df.iloc[0][date_column],
            errors="coerce"
        )

        if pd.notna(first_date):
            break

        cleaned_df = cleaned_df.iloc[1:].reset_index(drop=True)

    return cleaned_df

def interpolate_missing_values(
        df,
        columns,
        date_column="Date",
        method="time",
        limit_direction="both"
):
    """
    Interpolate missing values in selected columns.

    For method='time', the date column is temporarily used as the index.
    """
    interpolated_df = df.copy()

    interpolated_df[date_column] = pd.to_datetime(
        interpolated_df[date_column],
        errors="coerce"
    )

    interpolated_df = (
        interpolated_df
        .sort_values(date_column)
        .reset_index(drop=True)
    )

    missing_columns = [
        column
        for column in columns
        if column not in interpolated_df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Columns not found in DataFrame: {missing_columns}"
        )

    if method == "time":
        interpolated_df = interpolated_df.set_index(date_column)

        interpolated_df[columns] = (
            interpolated_df[columns]
            .interpolate(
                method="time",
                limit_direction=limit_direction
            )
        )

        interpolated_df = interpolated_df.reset_index()

    else:
        interpolated_df[columns] = (
            interpolated_df[columns]
            .interpolate(
                method=method,
                limit_direction=limit_direction
            )
        )

    return interpolated_df

def detect_iqr_outliers(
        df,
        columns,
        multiplier=1.5,
        add_flags=True
):
    """
    Detect outliers using the IQR method.

    Returns
    -------
    flagged_df : pandas.DataFrame
        Dataset with optional outlier flags.
    summary_df : pandas.DataFrame
        IQR bounds and outlier counts for each column.
    """
    flagged_df = df.copy()
    summary = []

    for column in columns:
        if column not in flagged_df.columns:
            raise KeyError(f"Column not found: {column}")

        series = pd.to_numeric(
            flagged_df[column],
            errors="coerce"
        )

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        outlier_mask = (
                (series < lower_bound)
                | (series > upper_bound)
        )

        if add_flags:
            flagged_df[f"{column}_Outlier"] = outlier_mask

        summary.append({
            "Feature": column,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Lower_Bound": lower_bound,
            "Upper_Bound": upper_bound,
            "Outlier_Count": int(outlier_mask.sum())
        })

    summary_df = pd.DataFrame(summary)

    return flagged_df, summary_df

def align_time_series(
        base_df,
        other_dfs,
        date_column="Date",
        how="left",
        fill_columns=None,
        fill_method="ffill"
):
    """
    Align multiple time-series datasets to a base date index.
    """
    aligned_df = base_df.copy()

    aligned_df[date_column] = pd.to_datetime(
        aligned_df[date_column],
        errors="coerce"
    )

    aligned_df = (
        aligned_df
        .dropna(subset=[date_column])
        .sort_values(date_column)
        .drop_duplicates(subset=[date_column], keep="last")
        .reset_index(drop=True)
    )

    for other_df in other_dfs:
        current_df = other_df.copy()

        current_df[date_column] = pd.to_datetime(
            current_df[date_column],
            errors="coerce"
        )

        current_df = (
            current_df
            .dropna(subset=[date_column])
            .sort_values(date_column)
            .drop_duplicates(subset=[date_column], keep="last")
        )

        aligned_df = aligned_df.merge(
            current_df,
            on=date_column,
            how=how
        )

    aligned_df = (
        aligned_df
        .sort_values(date_column)
        .reset_index(drop=True)
    )

    if fill_columns:
        missing_columns = [
            column
            for column in fill_columns
            if column not in aligned_df.columns
        ]

        if missing_columns:
            raise KeyError(
                f"Columns not found after merging: {missing_columns}"
            )

        if fill_method == "ffill":
            aligned_df[fill_columns] = (
                aligned_df[fill_columns].ffill()
            )

        elif fill_method == "bfill":
            aligned_df[fill_columns] = (
                aligned_df[fill_columns].bfill()
            )

        elif fill_method == "both":
            aligned_df[fill_columns] = (
                aligned_df[fill_columns]
                .ffill()
                .bfill()
            )

        else:
            raise ValueError(
                "fill_method must be 'ffill', 'bfill', or 'both'."
            )

    return aligned_df

def engineer_features(
        df,
        price_column="Close",
        vix_column="VIX_Close",
        rate_column="Treasury_Rate",
        trading_days=252
):
    """
    Construct financial features from cleaned and aligned market data.
    """
    feature_df = df.copy()

    required_columns = [
        price_column,
        vix_column,
        rate_column
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in feature_df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Required columns not found: {missing_columns}"
        )

    # Daily percentage return of JPM closing price
    feature_df["Daily_Return"] = (
        feature_df[price_column].pct_change()
    )

    # Log return of JPM closing price
    feature_df["Log_Return"] = np.log(
        feature_df[price_column]
        / feature_df[price_column].shift(1)
    )

    # Annualized 20-day rolling historical volatility
    feature_df["Rolling_Volatility_20D"] = (
            feature_df["Daily_Return"]
            .rolling(window=20)
            .std()
            * np.sqrt(trading_days)
    )

    # 20-day moving average
    feature_df["Moving_Average_20D"] = (
        feature_df[price_column]
        .rolling(window=20)
        .mean()
    )

    # 50-day moving average
    feature_df["Moving_Average_50D"] = (
        feature_df[price_column]
        .rolling(window=50)
        .mean()
    )

    # 20-day percentage price momentum
    feature_df["Price_Momentum_20D"] = (
        feature_df[price_column]
        .pct_change(periods=20)
    )

    # Daily change in Treasury rate
    feature_df["Interest_Rate_Momentum"] = (
        feature_df[rate_column]
        .diff()
    )

    # Daily percentage return of VIX
    feature_df["VIX_Return"] = (
        feature_df[vix_column]
        .pct_change()
    )

    # 20-day rolling correlation between JPM and VIX returns
    feature_df["VIX_JPM_Correlation_20D"] = (
        feature_df["Daily_Return"]
        .rolling(window=20)
        .corr(feature_df["VIX_Return"])
    )

    return feature_df

def clean_jpm_data(file_path):
    """
    Load and clean the Yahoo Finance JPM dataset.

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path to the JPM Yahoo Finance CSV file.

    Returns
    -------
    pandas.DataFrame
        Cleaned JPM market dataset.
    """
    jpm = pd.read_csv(file_path)

    # Standardize Yahoo Finance column names
    jpm.columns = [
        "Date",
        "Close",
        "High",
        "Low",
        "Open",
        "Volume"
    ]

    # Remove Yahoo Finance metadata rows
    jpm = remove_invalid_leading_rows(
        jpm,
        date_column="Date"
    )

    # Convert date column
    jpm["Date"] = pd.to_datetime(
        jpm["Date"],
        errors="coerce"
    )

    # Convert market columns to numeric values
    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    jpm[numeric_columns] = jpm[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Remove invalid dates, sort chronologically, and remove duplicates
    jpm = (
        jpm
        .dropna(subset=["Date"])
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )

    # Interpolate missing JPM price observations
    price_columns = [
        "Open",
        "High",
        "Low",
        "Close"
    ]

    if jpm[price_columns].isna().any().any():
        jpm = interpolate_missing_values(
            jpm,
            columns=price_columns,
            date_column="Date"
        )

    # Flag unusual trading volume observations
    jpm, outlier_summary = detect_iqr_outliers(
        jpm,
        columns=["Volume"]
    )

    return jpm, outlier_summary

def clean_vix_data(file_path):
    """
    Load and clean the Yahoo Finance VIX dataset.

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path to the VIX CSV file.

    Returns
    -------
    pandas.DataFrame
        Cleaned VIX dataset.
    """
    vix = pd.read_csv(file_path)

    # Standardize Yahoo Finance VIX column names
    vix.columns = [
        "Date",
        "VIX_Close",
        "VIX_High",
        "VIX_Low",
        "VIX_Open",
        "VIX_Volume"
    ]

    # Remove Yahoo Finance metadata rows
    vix = remove_invalid_leading_rows(
        vix,
        date_column="Date"
    )

    # Convert date column
    vix["Date"] = pd.to_datetime(
        vix["Date"],
        errors="coerce"
    )

    # Convert market columns to numeric values
    numeric_columns = [
        "VIX_Open",
        "VIX_High",
        "VIX_Low",
        "VIX_Close",
        "VIX_Volume"
    ]

    vix[numeric_columns] = vix[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Remove invalid dates, sort chronologically, and remove duplicates
    vix = (
        vix
        .dropna(subset=["Date"])
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )

    # Interpolate missing VIX price observations
    price_columns = [
        "VIX_Open",
        "VIX_High",
        "VIX_Low",
        "VIX_Close"
    ]

    if vix[price_columns].isna().any().any():
        vix = interpolate_missing_values(
            vix,
            columns=price_columns,
            date_column="Date"
        )

    # Flag unusual VIX closing values
    vix, outlier_summary = detect_iqr_outliers(
        vix,
        columns=["VIX_Close"]
    )

    return vix, outlier_summary

def clean_treasury_data(file_path):
    """
    Load and clean the U.S. Treasury rate dataset.

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path to the Treasury rate CSV file.

    Returns
    -------
    pandas.DataFrame
        Cleaned Treasury rate dataset.
    """
    treasury = pd.read_csv(file_path)

    # Standardize Treasury column names
    treasury.columns = [
        "Date",
        "Treasury_Rate"
    ]

    # Convert date and rate columns
    treasury["Date"] = pd.to_datetime(
        treasury["Date"],
        errors="coerce"
    )

    treasury["Treasury_Rate"] = pd.to_numeric(
        treasury["Treasury_Rate"],
        errors="coerce"
    )

    # Remove invalid dates, sort chronologically, and remove duplicates
    treasury = (
        treasury
        .dropna(subset=["Date"])
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )

    # Interpolate missing Treasury rate observations
    if treasury[["Treasury_Rate"]].isna().any().any():
        treasury = interpolate_missing_values(
            treasury,
            columns=["Treasury_Rate"],
            date_column="Date"
        )

    # Flag unusual Treasury rate observations
    treasury, outlier_summary = detect_iqr_outliers(
        treasury,
        columns=["Treasury_Rate"]
    )

    return treasury, outlier_summary

def preprocess_pipeline():
    """
    Execute the complete financial data preprocessing pipeline.

    The pipeline loads raw datasets, cleans each data source,
    aligns time series, constructs financial features, detects
    feature outliers, and exports the processed dataset.
    """
    print("Starting preprocessing pipeline...")

    # Load and clean the raw datasets
    jpm, jpm_outlier_summary = clean_jpm_data(
        DATA_DIR / "JPM_Yahoo_2018_2024.csv"
    )

    vix, vix_outlier_summary = clean_vix_data(
        DATA_DIR / "VIX_2018_2024.csv"
    )

    treasury, treasury_outlier_summary = clean_treasury_data(
        DATA_DIR / "Treasury_2018_2024.csv"
    )

    print("Raw datasets cleaned successfully.")

    # Keep only the VIX columns required for merging
    vix_for_merge = vix[
        [
            "Date",
            "VIX_Close",
            "VIX_Close_Outlier"
        ]
    ].copy()

    # Keep only the Treasury columns required for merging
    treasury_for_merge = treasury[
        [
            "Date",
            "Treasury_Rate",
            "Treasury_Rate_Outlier"
        ]
    ].copy()

    # Align VIX and Treasury data to the JPM trading calendar
    market_data = align_time_series(
        base_df=jpm,
        other_dfs=[
            vix_for_merge,
            treasury_for_merge
        ],
        date_column="Date",
        how="left",
        fill_columns=[
            "VIX_Close",
            "Treasury_Rate"
        ],
        fill_method="both"
    )

    print("Time series aligned successfully.")

    # Construct financial features
    market_data = engineer_features(
        market_data
    )

    # Detect outliers in selected engineered features
    market_data, feature_outlier_summary = detect_iqr_outliers(
        market_data,
        columns=[
            "Daily_Return",
            "Rolling_Volatility_20D",
            "Price_Momentum_20D"
        ]
    )

    print("Financial features generated successfully.")

    # Create output directory if it does not already exist
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save the complete processed dataset
    market_data.to_csv(
        OUTPUT_DIR / "market_data_processed.csv",
        index=False
    )

    # Save outlier summary tables
    jpm_outlier_summary.to_csv(
        OUTPUT_DIR / "jpm_outlier_summary.csv",
        index=False
    )

    vix_outlier_summary.to_csv(
        OUTPUT_DIR / "vix_outlier_summary.csv",
        index=False
    )

    treasury_outlier_summary.to_csv(
        OUTPUT_DIR / "treasury_outlier_summary.csv",
        index=False
    )

    feature_outlier_summary.to_csv(
        OUTPUT_DIR / "feature_outlier_summary.csv",
        index=False
    )

    print(
        "Preprocessing completed successfully. "
        f"Processed files were saved to: {OUTPUT_DIR.resolve()}"
    )

    return market_data

if __name__ == "__main__":
    preprocess_pipeline()
from typing import Annotated, Dict, Tuple, Union

import os
import tempfile

from langchain.tools import tool

from langgraph.prebuilt import InjectedState

from src.tools.dataframe import get_dataframe_summary


@tool(response_format="content")
def explain_data(
    data_raw: Annotated[dict, InjectedState("data_raw")],
    n_sample: int = 30,
    skip_stats: bool = False,
):
    """
    Tool: explain_data
    Description:
        Provides an extensive, narrative summary of a DataFrame including its shape, column types,
        missing value percentages, unique counts, sample rows, and (if not skipped) descriptive stats/info.

    Parameters:
        data_raw (dict): Raw data.
        n_sample (int, default=30): Number of rows to display.
        skip_stats (bool, default=False): If True, omit descriptive stats/info.

    LLM Guidance:
        Use when a detailed, human-readable explanation is needed—i.e., a full overview is preferred over a concise numerical summary.

    Returns:
        str: Detailed DataFrame summary.
    """
    print("    * Tool: explain_data")
    import pandas as pd

    result = get_dataframe_summary(
        pd.DataFrame(data_raw), n_sample=n_sample, skip_stats=skip_stats
    )

    return result


@tool(response_format="content_and_artifact")
def describe_dataset(
    data_raw: Annotated[dict, InjectedState("data_raw")],
) -> Tuple[str, Dict]:
    """
    Tool: describe_dataset
    Description:
        Compute and return summary statistics for the dataset using pandas' describe() method.
        The tool provides both a textual summary and a structured artifact (a dictionary) for further processing.

    Parameters:
    -----------
    data_raw : dict
        The raw data in dictionary format.

    LLM Selection Guidance:
    ------------------------
    Use this tool when:
      - The request emphasizes numerical descriptive statistics (e.g., count, mean, std, min, quartiles, max).
      - The user needs a concise statistical snapshot rather than a detailed narrative.
      - Both a brief text explanation and a structured data artifact (for downstream tasks) are required.

    Returns:
    -------
    Tuple[str, Dict]:
        - content: A textual summary indicating that summary statistics have been computed.
        - artifact: A dictionary (derived from DataFrame.describe()) containing detailed statistical measures.
    """
    print("    * Tool: describe_dataset")
    import pandas as pd

    df = pd.DataFrame(data_raw)
    description_df = df.describe(include="all")
    content = "Summary statistics computed using pandas describe()."
    artifact = {"describe_df": description_df.to_dict()}
    return content, artifact


@tool(response_format="content_and_artifact")
def visualise_missing(
    data_raw: Annotated[dict, InjectedState("data_raw")], n_sample: int = None
) -> Tuple[str, Dict]:
    """
    Tool: visualise_missing
    Description:
        Missing value analysis using the missingno library. Generates a matrix plot, bar plot, and heatmap plot.

    Parameters:
    -----------
    data_raw : dict
        The raw data in dictionary format.
    n_sample : int, optional (default: None)
        The number of rows to sample from the dataset if it is large.

    Returns:
    -------
    Tuple[str, Dict]:
        content: A message describing the generated plots.
        artifact: A dict with keys 'matrix_plot', 'bar_plot', and 'heatmap_plot' each containing the
                  corresponding base64 encoded PNG image.
    """
    print("    * Tool: visualise_missing")

    try:
        import missingno as msno  # Ensure missingno is installed
    except ImportError:
        raise ImportError(
            "Please install the 'missingno' package to use this tool. pip install missingno"
        )

    import pandas as pd
    import base64
    from io import BytesIO
    import matplotlib.pyplot as plt

    # Create the DataFrame and sample if n_sample is provided.
    df = pd.DataFrame(data_raw)
    if n_sample is not None:
        df = df.sample(n=n_sample, random_state=42)

    # Dictionary to store the base64 encoded images for each plot.
    encoded_plots = {}

    # Define a helper function to create a plot, save it, and encode it.
    def create_and_encode_plot(plot_func, plot_name: str):
        plt.figure(figsize=(8, 6))
        # Call the missingno plotting function.
        plot_func(df)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # Create and encode the matrix plot.
    encoded_plots["matrix_plot"] = create_and_encode_plot(msno.matrix, "matrix")

    # Create and encode the bar plot.
    encoded_plots["bar_plot"] = create_and_encode_plot(msno.bar, "bar")

    # Create and encode the heatmap plot.
    encoded_plots["heatmap_plot"] = create_and_encode_plot(msno.heatmap, "heatmap")

    content = (
        "Missing data visualisations (matrix, bar, and heatmap) have been generated."
    )
    artifact = encoded_plots
    return content, artifact


@tool(response_format="content_and_artifact")
def generate_correlation_funnel(
    data_raw: Annotated[dict, InjectedState("data_raw")],
    target: str,
    target_bin_index: Union[int, str] = -1,
    corr_method: str = "pearson",
    n_bins: int = 4,
    thresh_infreq: float = 0.01,
    name_infreq: str = "-OTHER",
) -> Tuple[str, Dict]:
    """
    Tool: generate_correlation_funnel
    Description:
        Correlation analysis using the correlation funnel method. The tool binarises the data and computes correlation versus a target column.

    Parameters:
    ----------
    target : str
        The base target column name (e.g., 'Member_Status'). The tool will look for columns that begin
        with this string followed by '__' (e.g., 'Member_Status__Gold', 'Member_Status__Platinum').
    target_bin_index : int or str, default -1
        If an integer, selects the target level by position from the matching columns.
        If a string (e.g., "Yes"), attempts to match to the suffix of a column name
        (i.e., 'target__Yes').
    corr_method : str
        The correlation method ('pearson', 'kendall', or 'spearman'). Default is 'pearson'.
    n_bins : int
        The number of bins to use for binarisation. Default is 4.
    thresh_infreq : float
        The threshold for infrequent levels. Default is 0.01.
    name_infreq : str
        The name to use for infrequent levels. Default is '-OTHER'.
    """
    print("    * Tool: generate_correlation_funnel")
    try:
        import pytimetk as tk
    except ImportError:
        raise ImportError(
            "Please install the 'pytimetk' package to use this tool. pip install pytimetk"
        )
    import pandas as pd
    import base64
    from io import BytesIO
    import matplotlib.pyplot as plt

    # Create the DataFrame
    df = pd.DataFrame(data_raw)

    # Find target columns that match the pattern
    target_columns = [col for col in df.columns if col.startswith(f"{target}__")]

    if not target_columns:
        raise ValueError(f"No columns found starting with '{target}__'")

    # Select the target column based on target_bin_index
    if isinstance(target_bin_index, int):
        if target_bin_index == -1:
            target_bin_index = len(target_columns) - 1
        target_column = target_columns[target_bin_index]
    else:
        # String matching
        matching_cols = [col for col in target_columns if col.endswith(f"__{target_bin_index}")]
        if not matching_cols:
            raise ValueError(f"No column found ending with '__{target_bin_index}'")
        target_column = matching_cols[0]

    # Binarise the data
    df_binarised = tk.binarize(
        df,
        n_bins=n_bins,
        thresh_infreq=thresh_infreq,
        name_infreq=name_infreq
    )

    # Calculate correlations
    correlations = df_binarised.corr(method=corr_method)[target_column].sort_values(ascending=False)

    # Create correlation funnel plot
    plt.figure(figsize=(10, 8))
    correlations.drop(target_column).plot(kind='barh')
    plt.title(f'Correlation Funnel - {target_column}')
    plt.xlabel('Correlation')
    plt.tight_layout()

    # Save plot to base64
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    content = f"Correlation funnel analysis completed for target: {target_column}"
    artifact = {
        "correlation_plot": plot_base64,
        "correlations": correlations.to_dict(),
        "target_column": target_column
    }
    return content, artifact


@tool(response_format="content_and_artifact")
def generate_sweetviz_report(
    data_raw: Annotated[dict, InjectedState("data_raw")],
    target: str = None,
    report_name: str = "sweetviz_report.html",
    report_directory: str = None,
    open_browser: bool = False,
) -> Tuple[str, Dict]:
    """
    Tool: generate_sweetviz_report
    Description:
        Generate a comprehensive EDA report using Sweetviz library.

    Parameters:
    -----------
    data_raw : dict
        The raw data in dictionary format.
    target : str, optional
        The target column for analysis.
    report_name : str
        Name of the HTML report file.
    report_directory : str, optional
        Directory to save the report. If None, uses temp directory.
    open_browser : bool
        Whether to open the report in browser.

    Returns:
    -------
    Tuple[str, Dict]:
        content: A message describing the generated report.
        artifact: A dict with report path and summary information.
    """
    print("    * Tool: generate_sweetviz_report")

    try:
        import sweetviz as sv
    except ImportError:
        raise ImportError(
            "Please install the 'sweetviz' package to use this tool. pip install sweetviz"
        )

    import pandas as pd

    # Create the DataFrame
    df = pd.DataFrame(data_raw)

    # Set up report directory
    if report_directory is None:
        report_directory = tempfile.gettempdir()

    report_path = os.path.join(report_directory, report_name)

    # Generate the report
    if target and target in df.columns:
        report = sv.analyze(df, target_feat=target)
    else:
        report = sv.analyze(df)

    # Save the report
    report.show_html(filepath=report_path, open_browser=open_browser)

    content = f"Sweetviz EDA report generated and saved to: {report_path}"
    artifact = {
        "report_path": report_path,
        "target_column": target,
        "dataset_shape": df.shape
    }
    return content, artifact


@tool(response_format="content_and_artifact")
def generate_dtale_report(
    data_raw: Annotated[dict, InjectedState("data_raw")],
    host: str = "localhost",
    port: int = 40000,
    open_browser: bool = False,
) -> Tuple[str, Dict]:
    """
    Tool: generate_dtale_report
    Description:
        Generate an interactive data exploration interface using D-Tale.

    Parameters:
    -----------
    data_raw : dict
        The raw data in dictionary format.
    host : str
        Host address for the D-Tale server.
    port : int
        Port number for the D-Tale server.
    open_browser : bool
        Whether to open the interface in browser.

    Returns:
    -------
    Tuple[str, Dict]:
        content: A message describing the generated interface.
        artifact: A dict with server information.
    """
    print("    * Tool: generate_dtale_report")

    try:
        import dtale
    except ImportError:
        raise ImportError(
            "Please install the 'dtale' package to use this tool. pip install dtale"
        )

    import pandas as pd

    # Create the DataFrame
    df = pd.DataFrame(data_raw)

    # Start D-Tale
    d = dtale.show(df, host=host, port=port, open_browser=open_browser)

    content = f"D-Tale interactive interface started at: {d._url}"
    artifact = {
        "url": d._url,
        "host": host,
        "port": port,
        "dataset_shape": df.shape
    }
    return content, artifact 
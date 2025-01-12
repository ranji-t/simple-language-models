import marimo

__generated_with = "0.10.12"
app = marimo.App(width="full", app_title="Bigram Graph")


@app.cell
def _():
    # Third pary imports
    import nltk
    import marimo as mo
    import polars as pl
    import plotly.express as px
    import networkx as nx
    return mo, nltk, nx, pl, px


@app.cell
def _():
    # Text file path
    file_path: str = r"../data/text_files/The Sword of Destiny.txt"

    # Read text file
    with open(file_path, "r") as f:
        text = f.read()

    # Display Text
    text
    return f, file_path, text


@app.cell
def _():
    # # Down Load data
    # nltk.download("punkt", download_dir="../data/nltk_data")
    return


@app.cell
def _(text):
    # Third party imports
    import re
    from string import punctuation
    from nltk.util import bigrams
    from nltk.tokenize import word_tokenize, sent_tokenize


    # Data Copy
    text_refined: str = text

    # Punctuation modification
    punctuation_list = [char for char in punctuation]
    punctuation_list.extend(["ï", "»", "¿"])
    punctuation_list.remove("'")
    punctuation_list.remove(",")

    # Rempve Puncutatoins
    for punct in punctuation_list:
        text_refined = text_refined.replace(punct, " ")

    # Replace "'" with ""
    text_refined = text_refined.replace("'", "")
    text_refined = text_refined.replace(",", "")

    # Replace white spaces
    text_refined = re.sub(r"\s+", " ", text_refined).lower()

    # Tokenlize the data
    tokens = word_tokenize(text_refined)

    # Convert to Bigram
    bigram_list = list(bigrams(tokens))

    # Display Bigram list
    bigram_list[:10]
    return (
        bigram_list,
        bigrams,
        punct,
        punctuation,
        punctuation_list,
        re,
        sent_tokenize,
        text_refined,
        tokens,
        word_tokenize,
    )


@app.cell
def _(bigram_list, pl):
    # Segnificance of Frequency
    FREQUECY_CUTOFF = 2

    # Create Frequency Table
    freq_table = (
        pl.LazyFrame(
            bigram_list,
            schema=[("source", pl.String), ("target", pl.String)],
            orient="row",
        )
        .with_columns(pl.lit(1).cast(pl.UInt64).alias("weight"))
        .group_by("source", "target")
        .agg(pl.col("weight").sum())
        .sort("weight", descending=False)
        .filter(pl.col("weight") > FREQUECY_CUTOFF)
        .collect(streaming=True)
    )

    # Display Data
    freq_table
    return FREQUECY_CUTOFF, freq_table


@app.cell
def _(freq_table, pl, px):
    px.histogram(
        data_frame=freq_table.select(pl.col("weight")),
        x="weight",
    )
    return


@app.cell
def _(freq_table, nx):
    # Directed Graph
    d_graph = nx.from_pandas_edgelist(
        df=freq_table.to_pandas(),
        source="source",
        target="target",
        edge_attr=["weight"],
        create_using=nx.DiGraph(),
    )
    return (d_graph,)


@app.cell
def _(d_graph, pl):
    # Max iterations
    MAX_ITER = 10
    RELATIVE_PROBA = 0.50

    # Sentence List
    sentence_list = []

    # Define the node for which you want to find the maximum weight edge
    start_node = ["horse"]
    sentence_list.extend(start_node)

    for _ in range(MAX_ITER):
        # Get all outgoing edges from the start node with their weights
        outgoing_edges = d_graph.out_edges(start_node, data=True)

        # No follow up word
        if len(outgoing_edges) == 0:
            break

        # The next Word
        next_word = (
            pl.DataFrame(
                list(outgoing_edges),
                orient="row",
                schema=[
                    ("source", pl.String),
                    ("target", pl.String),
                    ("weight", pl.Struct({"weight": pl.UInt64})),
                ],
            )
            .with_columns(pl.col("weight").struct.field("weight"))
            .with_columns(
                pl.col("weight").alias("relative_proba") / pl.col("weight").max()
            )
            .sort("weight", descending=True)
            .filter(pl.col("relative_proba") >= RELATIVE_PROBA)
            .sample(1)
            .select("target")
            .item(0, 0)
        )
        # Reset Start Node
        start_node = [next_word]
        sentence_list.extend(start_node)

    # Form A sentence
    sentence = " ".join(sentence_list)

    # Print Output
    print(sentence)
    return (
        MAX_ITER,
        RELATIVE_PROBA,
        next_word,
        outgoing_edges,
        sentence,
        sentence_list,
        start_node,
    )


if __name__ == "__main__":
    app.run()

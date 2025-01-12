import marimo

__generated_with = "0.10.12"
app = marimo.App(width="full", app_title="Bigram Graph")


@app.cell
def _(mo):
    mo.md("""# **Sentence Generation System: Bigram Graphs**""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # Segnificance of Frequency
    FREQUENCY_CUTOFF = 2
    # Max iterations
    MAX_ITER = 10
    RELATIVE_PROBA = 0.50
    return FREQUENCY_CUTOFF, MAX_ITER, RELATIVE_PROBA


@app.cell
def _(FREQUENCY_CUTOFF, MAX_ITER, RELATIVE_PROBA):
    # Standard Imports
    import re

    # Third party imports
    import polars as pl
    import networkx as nx
    import plotly.express as px
    from string import punctuation
    from nltk.util import bigrams
    from nltk.tokenize import word_tokenize, sent_tokenize


    def read_text_file(
        file_path: str = r"../data/text_files/The Sword of Destiny.txt",
    ):
        # Read text file
        with open(file_path, "r") as f:
            text = f.read()

        # Return Text File
        return text


    def text_processing(text: str) -> list[tuple[str, str]]:
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

        # Return Text Bigram
        return bigram_list


    def frequecy_table(
        bigram_list: list[tuple[str, str]],
        frequency_cutoff: int = FREQUENCY_CUTOFF,
    ) -> pl.DataFrame:
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
            .filter(pl.col("weight") > frequency_cutoff)
            .collect(streaming=True)
        )

        # Display Data
        return freq_table


    def generate_graph(freq_table: pl.DataFrame) -> nx.DiGraph:
        # Directed Graph
        d_graph = nx.from_pandas_edgelist(
            df=freq_table.to_pandas(),
            source="source",
            target="target",
            edge_attr=["weight"],
            create_using=nx.DiGraph(),
        )

        return d_graph


    def show_histogram(freq_table: pl.DataFrame) -> None:
        px.histogram(
            data_frame=freq_table.select(pl.col("weight")),
            x="weight",
        ).show()


    def generate_sentence(
        first_word: str,
        d_graph: nx.DiGraph,
        relative_proba: float = RELATIVE_PROBA,
        max_iter: int = MAX_ITER,
    ) -> str:
        # Sentence List
        sentence_list = []

        # Define the node for which you want to find the maximum weight edge
        start_node = [first_word.lower().strip()]
        sentence_list.extend(start_node)

        for _ in range(max_iter):
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
                    pl.col("weight").alias("relative_proba")
                    / pl.col("weight").max()
                )
                .sort("weight", descending=True)
                .filter(pl.col("relative_proba") >= relative_proba)
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
        return sentence
    return (
        bigrams,
        frequecy_table,
        generate_graph,
        generate_sentence,
        nx,
        pl,
        punctuation,
        px,
        re,
        read_text_file,
        sent_tokenize,
        show_histogram,
        text_processing,
        word_tokenize,
    )


@app.cell
def _():
    # # Down Load data
    # nltk.download("punkt", download_dir="../data/nltk_data")
    return


@app.cell
def _(mo):
    mo.md("""Frequecy CutOff:""")
    return


@app.cell
def _(FREQUENCY_CUTOFF, mo):
    frequency_cutoff = mo.ui.slider(
        start=0,
        stop=100,
        step=1,
        value=FREQUENCY_CUTOFF,
        debounce=True,
        show_value=True,
    )
    frequency_cutoff
    return (frequency_cutoff,)


@app.cell
def _(mo):
    mo.md("""Relative Probabality:""")
    return


@app.cell
def _(RELATIVE_PROBA, mo):
    relative_proba = mo.ui.slider(
        start=0.5,
        stop=1,
        step=0.01,
        value=RELATIVE_PROBA,
        debounce=True,
        show_value=True,
    )
    relative_proba
    return (relative_proba,)


@app.cell
def _(mo):
    mo.md("""Max Iteration:""")
    return


@app.cell
def _(MAX_ITER, mo):
    max_iter = mo.ui.slider(
        start=1,
        stop=100,
        step=1,
        value=MAX_ITER,
        debounce=True,
        show_value=True,
    )
    max_iter
    return (max_iter,)


@app.cell
def _(mo):
    mo.md("""Initial Word Seed:""")
    return


@app.cell
def _(mo):
    first_word = mo.ui.text(value="lord")
    first_word
    return (first_word,)


@app.cell
def _(
    first_word,
    frequecy_table,
    frequency_cutoff,
    generate_graph,
    generate_sentence,
    max_iter,
    read_text_file,
    relative_proba,
    text_processing,
):
    # Read Text
    text = read_text_file(file_path=r"../data/text_files/The Sword of Destiny.txt")

    # Process Raw data in to Bigrams
    bigram_list = text_processing(text=text)

    # Create Frequency Table
    freq_table = frequecy_table(
        bigram_list=bigram_list, frequency_cutoff=frequency_cutoff.value
    )

    # Show HistoGram
    # show_histogram(freq_table=freq_table)

    # Generate Directional Graph
    d_graph = generate_graph(freq_table=freq_table)

    # Generate sentence
    senetence = generate_sentence(
        first_word=first_word.value,
        d_graph=d_graph,
        relative_proba=relative_proba.value,
        max_iter=max_iter.value,
    )
    return bigram_list, d_graph, freq_table, senetence, text


@app.cell
def _(mo):
    # Print sentence
    mo.md("## The Output")
    return


@app.cell
def _(mo, senetence):
    mo.md(f"> `{senetence}`")
    return


if __name__ == "__main__":
    app.run()

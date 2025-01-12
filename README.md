# Bigram Graph Language Model

This project implements a simple language model using bigrams, which are pairs of consecutive words from a given text. The model visualizes bigram relationships using graphs and generates sentences based on the most probable transitions between words.

---

## Features
- **Bigram Generation**: Tokenizes the text and generates bigrams.
- **Bigram Frequency Filtering**: Filters bigrams based on a frequency threshold.
- **Graph Representation**: Constructs a directed graph using `NetworkX` to represent bigram relationships.
- **Sentence Generation**: Generates sentences based on bigram probabilities.
- **Interactive Visualization**: Provides insights into bigram frequencies using Plotly histograms.

---

## Directory Structure
```
├── .subs/                # Submodules or dependencies
├── .venv/                # Python virtual environment
├── data/                 # Contains input data for the model
│   ├── nltk_data/        # NLTK data files
│   └── text_files/       # Text files used for processing (e.g., The Sword of Destiny.txt)
├── models/               # Directory for model-related artifacts (currently empty)
├── src/                  # Source code for the application
│   ├── modules/          # Additional modules for application logic
│   ├── note.ipynb        # Jupyter Notebook for exploration or debugging
│   └── note.py           # Main Python script for the application
├── test/                 # Testing and development utilities
├── .dockerignore         # Docker ignore file
├── .gitignore            # Git ignore file
├── compose.yaml          # Docker Compose configuration (optional)
├── Dockerfile            # Docker configuration file
├── README.Docker.md      # Additional Docker-specific documentation
├── README.md             # Main project documentation
├── requirements.txt      # Python dependencies file
```

---

## Installation and Usage

### Prerequisites
- Python 3.12 or later
- Docker (optional for containerized execution)

### Steps to Run the Project Locally
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the input text file is placed in `data/text_files/`.

4. Run the application:
   ```bash
   python src/note.py
   ```

---

## Running with Docker

To run the project using Docker, use the provided `Dockerfile`:

### Build the Docker Image
```bash
docker build -t bigram-graph-app .
```

### Run the Container
```bash
docker run -p 2719:2719 bigram-graph-app
```

The application will be accessible at `http://127.0.0.1:2719`.

For more details, refer to `README.Docker.md`.

---

## Dependencies

The project relies on the following Python libraries (specified in `requirements.txt`):
- `ipykernel==6.29.5`
- `marimo==0.10.12`
- `matplotlib==3.10.0`
- `nbformat==5.10.4`
- `networkx==3.4.2`
- `nltk==3.9.1`
- `numpy==2.2.1`
- `pandas==2.2.3`
- `plotly==5.24.1`
- `polars==1.19.0`
- `pyarrow==18.1.0`
- `ruff==0.9.1`
- `seaborn==0.13.2`
- `scikit-learn==1.6.1`
- `tqdm==4.67.1`

Install all dependencies using:
```bash
pip install -r requirements.txt
```

---

## Project Workflow

1. **Text Processing**:
   - Reads a text file.
   - Cleans and tokenizes the text using `nltk`.
   - Generates bigrams using `nltk.util.bigrams`.

2. **Frequency Analysis**:
   - Filters bigrams based on a frequency threshold (`FREQUENCY_CUTOFF`).
   - Uses Polars for efficient data manipulation.

3. **Graph Construction**:
   - Constructs a directed graph from bigrams using `NetworkX`.
   - Calculates transition probabilities.

4. **Sentence Generation**:
   - Starts from a given word (e.g., `horse`) and generates sentences using the graph's most probable transitions.

5. **Visualization**:
   - Visualizes bigram frequencies using Plotly.

---

## Example Usage

### Input
A text file placed in `data/text_files/` (e.g., `The Sword of Destiny.txt`).

### Output
- **Filtered Bigram Frequency Table**:
  ```
  source       target       weight
  -------      -------      ------
  'horse'      'ran'        5
  'ran'        'away'       3
  ...
  ```
- **Generated Sentence**:
  ```
  horse ran away into the woods.
  ```

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Acknowledgments

- The project uses the [NLTK library](https://www.nltk.org/) for natural language processing.
- Graph construction and visualization are powered by [NetworkX](https://networkx.org/) and [Plotly](https://plotly.com/).
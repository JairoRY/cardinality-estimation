# Cardinality Estimation Assignment

This project evaluates and compares the performance of the **HyperLogLog** and **Recordinality** algorithms for estimating the number of unique elements in large data streams. It features a C++ implementation for performance-critical estimation and a Python suite for analytical visualization.

---

### Requirements

To build and run this project, the following tools and libraries are required:

* **C++ Compiler**: A compiler supporting at least the C++11 standard (e.g., `g++` or `clang++`).
* **Python 3.x**: Required for parsing output and generating plots.
* **Python Libraries**:
  * `matplotlib`: For generating comparative charts.
  * `numpy`: Used for data manipulation during plotting.

---

### Compilation

The core logic is contained in `cardest.cpp`. It can be compiled using a standard C++ compiler with optimization flags for better performance:

```bash
g++ -O3 cardest.cpp -o cardest

```

*(The `-O3` flag is recommended to ensure the efficiency of high-volume stream processing.)*

---

### Execution

The project is executed in two stages: data generation and visualization.

1. **Run the Estimators**: Execute the compiled binary to process the datasets and synthetic streams. Redirect the output to a text file for analysis:
```bash
./cardest > output.txt

```

This executes three experiments: a general dataset comparison, a memory-impact study, and a robustness test against data skew.
2. **Generate Plots**: Use the provided Python script to process `output.txt` and generate the visualization files (e.g., `exp1_barplot.png`, `exp3_alphaplot.png`):
```bash
python3 plots.py

```

---

### Output

The execution produces several visual and textual outputs:

* **`output.txt`**: A detailed report containing the average estimates, relative errors, and standard errors for both HyperLogLog and Recordinality across all test cases.
* **Experiment 1 (Bar Plots)**: Visual comparison of estimation accuracy across various literary works and synthetic data.
* **Experiment 2 (Line Plots)**: Analysis of how increasing memory (registers  or sample size ) reduces the standard error towards theoretical benchmarks.
* **Experiment 3 (Line Plot)**: A chart demonstrating the robustness of both algorithms when processing skewed data streams (Zipfian ).

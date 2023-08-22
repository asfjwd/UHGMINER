# UHGMINER

<em> Utility-based Hypergraph Miner (UHGMINER) </em> algorithm mines high utility subhypergraph patterns from a quantitative labeled hypergraph database.

## Contents

- **datasets** - Folder containing the datasets that were used to conduct the experiments.
- **uhgminer_cwu.py** - Implementation of <em> UHGMINER-CWU </em>.
- **uhgminer_msu.py** - Implementation of <em> UHGMINER-MSU </em>.


## Requirements
**Python** version 3.9 or higher.

## Usage
```bash
python uhgminer_msu.py <dataset> <utility_threshold>
```

### Options for `dataset `

- ecommerce
- foodmart
- fruithut
- liquor
- bioinformatics
- computer_network
- computer_security
- data_mining
- distributed_computing
- machine_learning

### Options for `utility_threshold`

Any real value between 0 and 1.

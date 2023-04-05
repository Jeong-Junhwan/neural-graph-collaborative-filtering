# Neural Graph Collaborative Filtering (NGCF)

This repository contains a PyTorch implementation of the _Neural Graph Collaborative Filtering (NGCF)_ model, as proposed in the paper:

> Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, Tat-Seng Chua. "Neural Graph Collaborative Filtering" https://arxiv.org/abs/1905.08108

The implementation is owned by Jeong-Junhwan and includes a simple training process, but does not yet support hyperparameter tuning or dropout.

## Table of Contents

- [Requirements](#requirements)
- [Usage](#usage)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)

## Requirements

To install the required dependencies, run:

```bash
pip3 install -r requirements.txt
```

## Usage

To run the training process, simply execute the following command:

```bash
python3 train.py
```

Please note that this implementation currently does not support hyperparameter tuning or dropout. You are welcome to contribute by adding these features.

## Dataset

This implementation has been tested using the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/), which is a popular benchmark dataset for collaborative filtering models. It consists of 100,000 ratings from 1 to 5, given by 943 users on 1,682 movies.

To use the ML-100K dataset, please download it from the link above and place the unzipped files in a ml-100k/ directory within the project root.

## Contributing

Contributions are welcome! If you would like to improve this implementation, add new features or fix bugs, please feel free to submit a pull request.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more information.

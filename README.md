# LLMDataForge

Welcome to LLMDataForge, a cutting-edge framework that leverages large language models (LLMs) to generate high-quality datasets tailored to your needs. This README file will guide you through understanding the framework, setting it up, and generating your datasets.

## Introduction
LLMDataForge is designed to simplify the process of dataset generation by utilizing the power of large language models. Whether you need synthetic data for training machine learning models, testing, or any other purpose, LLMDataForge can help you generate diverse and realistic datasets efficiently.

### Capabilities
This framework is designed to tackle different types of datasets and filtering methods. It is still constantly being developed and for now supports just the text classification task. However the design is modular and simplistic so that adding new dataset types and filtering methods is easy.  

## Usage
To generate datasets using LLMDataForge, you need to run the app.py script with a specified configuration file. The configuration file determines the parameters and settings for dataset generation.

### installing dependencies

The project manages the dependencies with pdm. Run:

```
pdm install
```

### Running the Script
Use the following command to run the script:

```
python app.py --config_file {path to config file}
```

## Contributing
We welcome contributions from the community! If you'd like to contribute to LLMDataForge, please follow these steps:

1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Commit your changes (git commit -m 'Add new feature').
4. Push to the branch (git push origin feature-branch).
5. Open a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for more details.


# LazyPredict UI

A Visual Studio Code extension that provides an intuitive user interface for the [LazyPredict](https://github.com/shankarpandala/lazypredict) Python package. This extension allows data scientists to quickly run multiple machine learning models on their datasets without leaving their VS Code environment.

## Features

- **Interactive UI**: Easy-to-use interface for running LazyPredict models
- **Dataset Management**: Upload and select CSV, Excel, or other tabular data files
- **Model Selection**: Choose which models to include in your LazyPredict run
- **Results Visualization**: View model performance metrics in an organized display
- **Python Integration**: Seamlessly works with your existing Python environment

## Requirements

- Visual Studio Code 1.80.0 or higher
- Python 3.6 or higher
- LazyPredict Python package installed (`pip install lazypredict`)

## Installation

1. Install the extension from the Visual Studio Code Marketplace
2. Ensure you have the LazyPredict Python package installed in your environment

## Usage

1. Open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`)
2. Type "Show LazyPredict UI" and select the command
3. Follow the UI instructions to upload a dataset and run models

## Extension Settings

This extension contributes the following settings:

* `lazypredict.pythonPath`: Path to Python interpreter with LazyPredict installed
* `lazypredict.defaultDatasetDir`: Default directory to browse for datasets

## Known Issues

- Current version requires LazyPredict to be pre-installed in your Python environment

## Release Notes

### 0.0.1

- Initial release with basic UI functionality
- Support for dataset selection and running classification/regression models

## Development

This extension was developed using Test-Driven Development (TDD) practices. All features are backed by comprehensive test suites.

### Building from Source

1. Clone the repository
2. Run `npm install`
3. Run `npm test` to execute tests
4. Run `npm run compile` to build the extension
5. Press F5 in VS Code to launch the Extension Development Host

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Developed by [Shankar Pandala](https://github.com/shankarpandala)**

# TableExtractor-Advanced-PDF-Table-Extraction

![Python Version](https://img.shields.io/badge/python-3.6+-blue.svg)
![license](https://img.shields.io/badge/license-MIT-green.svg)

Transform your scaned PDFs into actionable data with our advanced PDF Table Extractor. Utilizing state-of-the-art OCR and AI techniques, this Python tool effortlessly converts PDF documents into editable text formats, identifies and extracts tables, and integrates with Hugging Face Hub for further text processing.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features

- **PDF to Image Conversion:** Transforms PDF pages into images, preparing them for table detection and extraction.
- **Advanced Table Detection:** Employs morphological transformations to detect tables within images.
- **OCR Text Extraction:** Leverages OCR technology to extract text from tables accurately.
- **AI-Powered Text Processing:** Cleans and formats extracted text, using AI models from Hugging Face Hub.
- **Structured Data Output:** Aggregates extracted data into a structured and usable format.

## Prerequisites

Ensure you have the following prerequisites installed on your machine:
- Python 3.6 or later
- OpenCV (`cv2`) for image processing
- NumPy for array manipulation
- PyTesseract for OCR capabilities
- pdf2image for converting PDF pages into images
- PIL (Python Imaging Library) for image operations
- Hugging Face Transformers for AI model integration

## Installation

1. **Python Packages:**

    ```bash
    pip install opencv-python numpy pytesseract pdf2image Pillow transformers
    ```

2. **System Dependencies:** (For Debian/Ubuntu)

    ```bash
    sudo apt-get install poppler-utils
    sudo apt-get install tesseract-ocr
    
    ```
    
## Usage

### Basic Usage

To start extracting tables from your PDF document, instantiate the `TableExtractor` and provide the path to your document:

```python
from table_extractor import TableExtractor

pdf_path = "path/to/your/document.pdf"
extractor = TableExtractor(pdf_path)
```
## Contributing
🌟 We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug 🐛
- Discussing the current state of the code 🗣
- Submitting a fix 🔨
- Proposing new features ✨
- Becoming a maintainer 🚀

When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change.

Please note we have a code of conduct; please follow it in all your interactions with the project.

### Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a build.
2. Update the README.md with details of changes to the interface, including new environment variables, exposed ports, useful file locations, and container parameters.
3. Increase the version numbers in any examples files and the README.md to the new version that this Pull Request would represent. The versioning scheme we use is [SemVer](http://semver.org/).
4. You may merge the Pull Request in once you have the sign-off of two other developers, or if you do not have permission to do that, you may request the second reviewer to merge it for you.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.







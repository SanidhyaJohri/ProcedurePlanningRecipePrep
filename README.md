# Advancing Procedure Planning of Recipe Preparation with Multimodal Text-Image Guidance

Query2Food-Planning (QFP) is an innovative approach that marries the zero-shot reasoning capabilities of Large Language Models (LLMs) with the creative prowess of text-to-image diffusion models. This project aims to revolutionize how users engage with recipe preparation by providing both textual and visual instructions that guide them through the cooking process.

## Overview

QFP is built on two core components:
- **Query-to-Prompt Bridge**: This component transforms user queries about recipes into detailed prompts that LLMs can understand and process, enabling the generation of precise textual instructions for recipe preparation.
- **Prompt-to-Image Bridge**: Leveraging state-of-the-art diffusion models, this bridge turns the textual instructions into vivid, instructional images that visually guide users through the cooking steps.

Our methodology not only simplifies the process of following a recipe but also enhances the cooking experience by providing a visual representation of each step, making cooking more accessible and enjoyable.

## Features

- **Zero-Shot Reasoning**: Utilizes the advanced capabilities of LLMs to understand and generate cooking instructions from user queries without needing explicit examples of similar queries.
- **Text-to-Image Generation**: Employs diffusion models fine-tuned on a specialized recipe dataset, producing high-quality images that correspond to the generated textual instructions.
- **Web Application**: A user-friendly web application that streamlines the entire recipe preparation process, from querying to cooking, with both textual and visual aids.

## Comparative Evaluation

We conducted a comprehensive evaluation of QFP against other open-source diffusion models. Our analysis focused on several key metrics to ensure the effectiveness and accuracy of our generated plans and images. The results underscore QFP's potential to enhance the culinary experience significantly.

### Prerequisites

Before you begin, ensure you have the following installed on your system:
- Git
- Python 3.6 or later

### Instructions 

1. **Clone the repository**

   Start by cloning the repository to your local machine. Open a terminal and run the following command:
   ```sh
   git clone https://github.com/SanidhyaJohri/ProcedurePlanningRecipePrep.git
   ```

   Navigate to the folder containing the files using the command:
   ```sh
   cd ProcedurePlanningRecipePrep
   ```

2. **Setup a Virtual Environment**
   It's recommended to create a virtual environment to keep the dependencies required by different projects separate. Use the following command to create a virtual environment named venv:

   **For Windows:**
   ```cmd
   python -m venv venv
   .\venv\Scripts\activate
   ```

   **For macOS and Linux:**
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install Dependencies**
   With your virtual environment activated, install all required dependencies by running:
   ```sh
   pip install -r requirements.txt
   ```

5. Run the Streamlit Application
   Once the dependencies are installed, you can run the Streamlit application using:
   ```sh
   streamlit run main.py
   ```

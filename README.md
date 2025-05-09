# Tokenizers and HuggingFace Models Colab

## Introduction
Welcome to the Tokenizers and HuggingFace Models Colab project. This notebook shows how to use tokenizers and pre-trained models from HuggingFace in Google Colab. This README explains how to set up Git, clone the repo, run the notebook, and manage changes with Git.

## Prerequisites
- A GitHub account  
- Git installed on your local machine  
- A Google account to use Colab  

## Repository Setup
1. Open your terminal or command prompt.  
2. Clone the repository:  
   ```bash
   git clone <your-repo-url>
   ```  
3. Change to the project directory:  
   ```bash
   cd <repo-folder>
   ```  

## Running the Colab Notebook
1. Go to Google Colab: https://colab.research.google.com  
2. Click **File > Open notebook**.  
3. Select the **GitHub** tab and paste your repository URL.  
4. Open `Tokenizers_HuggingFace_Models.ipynb`.  
5. Run all cells by clicking **Runtime > Run all**.  

## Using Git for Version Control
After making changes to the notebook:
1. Stage your changes:  
   ```bash
   git add Tokenizers_HuggingFace_Models.ipynb
   ```  
2. Commit your changes with a message:  
   ```bash
   git commit -m "Describe your changes here"
   ```  
3. Push to GitHub:  
   ```bash
   git push origin main
   ```  
   Replace `main` with your branch name if different.

## Branching and Collaboration
- Create a new branch for each feature or fix:  
  ```bash
  git checkout -b feature-name
  ```  
- After finishing work, push the branch:  
  ```bash
  git push origin feature-name
  ```  
- Open a pull request on GitHub to merge changes.

## Troubleshooting
- If you see authentication errors, set up SSH keys or use a personal access token.  
- To update the notebook:  
  ```bash
  git pull origin main
  ```

## License
This project is licensed under the [MIT License](LICENSE).

## Contact
For questions or support, open an issue on GitHub.

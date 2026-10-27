# Sentinel-Deepfake-Detection
Multimodal Deepfake Detection System using Video and Audio Analysis
## Trained AI Model

The trained Sentinel video detection model is not stored directly in the Git repository.

Download the trained model from the Sentinel AI Model v1.0 release.

After downloading, place the file here:

weights/best_model.pth


## Windows Installation and Troubleshooting

### Prerequisites
- Python
- Git for Windows
- Visual Studio Code (recommended)

### Setup
1. Clone the repository.
2. Open the project folder in VS Code.
3. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

4. Activate it in Command Prompt:

   ```cmd
   venv\Scripts\activate
   ```

5. Install the dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Model Setup
Download the trained model from the **Sentinel AI Model v1.0** GitHub Release. Place the downloaded file at:

`weights/best_model.pth`

The application may fail to start if the model file is missing.

### Troubleshooting
- **Missing Python module:** Confirm the virtual environment is active and install the required dependencies.
- **Missing model file:** Verify that `weights/best_model.pth` exists at the expected path.
- **Startup errors:** Review the complete terminal traceback before changing the code.
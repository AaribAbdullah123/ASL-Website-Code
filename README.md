# ASL-Website-Code
I'm afraid due to resource constraints I can only supply the code not the already trained model or dataset. It is with deep regret that I say this is all I can present Im sorry for the inconvenience.

The code for the website w/ instructions on how to set it up and get it running. The Frontend and flask implementation is vibe-coded using the Qoder IDE. While the actual optimizations and training is Partially Hand-coded.

# Introductory Knowledge
There are 4 Parts of this project, the frontend, the backend, and the training script and the data collection script, the frontend is the website, the backend handles all the inference and processing, and the training script trains the PyTorch LSTM..

# Requirements
any version of python above 3.8, Dependencies: torch torchvision mediapipe opencv-python numpy flask  *Keep in mind needs to be the latest version of mediapipe(automatically installs files)*

Hardware Requirements: 
Operating System: Windows 10 64Bit, Windows 11
Atleast a Pascal NVIDIA Card NOT AMD (The training/inference uses <1GB VRAM) If dGPU is not available the script automatically fallbacks to CPU-only training If Using macOS or linux, script automatically detects OS and accounts for the change in code.
CPU: Any modern CPU
RAM: 8GB
Storage: 30MB
OS Drivers fully updated
A Webcam

# Instructions
*Highest priority*
Run the Data collection script and follow the instructions make sure the images are accurate for each word and make sure the lighting and clothes and different objects are different. 

Then Just Copy each file and save them in your workspace
Run the training script and wait until it finishes
Run the backend and the website will automatically open

That's it, you're good to go

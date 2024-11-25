#!/bin/bash

# Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is required but not installed. Please install Python3 first."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first
echo "Installing PyTorch..."
# Get the correct PyTorch command for the system
if [[ "$(uname)" == "Darwin" ]]; then
    # macOS
    if [[ "$(uname -m)" == "arm64" ]]; then
        # M1/M2 Mac
        pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
    else
        # Intel Mac
        pip3 install torch torchvision torchaudio
    fi
else
    # Linux/Windows with CUDA
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

# Verify PyTorch installation and wait for it to be accessible
echo "Verifying PyTorch installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" || exit 1
sleep 2  # Give time for the installation to fully complete

# Install torch-scatter with the correct version
echo "Installing torch-scatter..."
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VERSION=$(python3 -c "import torch; print('cpu' if not torch.cuda.is_available() else 'cu' + torch.version.cuda.replace('.',''))")
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html

# Install other requirements
echo "Installing other requirements..."
pip install -r <(grep -v "torch\|torch-scatter" requirements.txt)

# Install development package in editable mode
echo "Installing nexus in development mode..."
pip install -e .

# Verify PyTorch installation and CUDA availability
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo "Setup complete! Activate the virtual environment with: source venv/bin/activate"
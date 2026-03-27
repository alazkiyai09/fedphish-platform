#!/bin/bash

# FedPhish Demo Dashboard - One-time Setup Script

echo "🚀 Setting up FedPhish Demo Dashboard..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo "✓ Python version: $PYTHON_VERSION"

# Setup backend
echo ""
echo "📦 Setting up backend..."
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

echo "✓ Backend setup complete"

# Setup frontend
echo ""
echo "📦 Setting up frontend..."
cd ../frontend

# Install Node.js if not present
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 20+ first."
    echo "   Visit: https://nodejs.org/"
    exit 1
fi

NODE_VERSION=$(node --version)
echo "✓ Node version: $NODE_VERSION"

# Install dependencies
npm install

echo "✓ Frontend setup complete"

# Create .env file if not exists
cd ..
if [ ! -f backend/.env ]; then
    cat > backend/.env << EOF
HOST=0.0.0.0
PORT=8001
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
EOF
    echo "✓ Created backend/.env"
fi

if [ ! -f frontend/.env ]; then
    cat > frontend/.env << EOF
VITE_WS_URL=ws://localhost:8001/ws/federation
EOF
    echo "✓ Created frontend/.env"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the dashboard:"
echo "  1. Backend:   cd backend && source venv/bin/activate && python -m app.main"
echo "  2. Frontend:  cd frontend && npm run dev"
echo ""
echo "Or use Docker:"
echo "  docker-compose -f docker/docker-compose.yml up --build"
echo ""
echo "Access at: http://localhost:5173"

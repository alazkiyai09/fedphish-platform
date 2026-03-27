#!/bin/bash

# FedPhish Demo Dashboard - Development Mode

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting FedPhish Dashboard (Dev Mode)${NC}\n"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Stopping services...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo -e "${GREEN}✓ Services stopped${NC}"
    exit
}

# Trap SIGINT and SIGTERM
trap cleanup SIGINT SIGTERM

# Start backend
echo -e "${BLUE}Starting backend server...${NC}"
cd backend
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Run ./scripts/setup.sh first${NC}"
    exit 1
fi

source venv/bin/activate
python -m app.main &
BACKEND_PID=$!
echo -e "${GREEN}✓ Backend running (PID: $BACKEND_PID)${NC}"

# Wait a bit for backend to start
sleep 2

# Start frontend
echo -e "${BLUE}Starting frontend server...${NC}"
cd ../frontend

if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}node_modules not found. Run npm install first${NC}"
    kill $BACKEND_PID
    exit 1
fi

npm run dev &
FRONTEND_PID=$!
echo -e "${GREEN}✓ Frontend running (PID: $FRONTEND_PID)${NC}"

echo ""
echo -e "${GREEN}✅ Both services started!${NC}\n"
echo -e "Backend:  http://localhost:8001"
echo -e "Frontend: http://localhost:5173"
echo -e "API Docs:  http://localhost:8001/docs"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop both services${NC}"

# Wait for any process to exit
wait -n

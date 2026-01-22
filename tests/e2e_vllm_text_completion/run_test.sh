#!/bin/bash
#
# Automated test runner for text completion end-to-end tests
#
# This script:
# 1. Starts the mock VLLM server in the background
# 2. Runs the text completion tests
# 3. Stops the server when done
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Text Completion End-to-End Test Runner"
echo "=========================================="

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Warning: Virtual environment not activated${NC}"
    echo "Activating virtual environment..."
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Check dependencies
echo ""
echo "Checking dependencies..."
python -c "import flask" 2>/dev/null || {
    echo -e "${RED}Flask not found. Installing...${NC}"
    uv pip install flask
}
python -c "import requests" 2>/dev/null || {
    echo -e "${RED}Requests not found. Installing...${NC}"
    uv pip install requests
}
echo -e "${GREEN}✓ Dependencies OK${NC}"

# Start mock server in background
echo ""
echo "Starting mock VLLM server..."
python "$SCRIPT_DIR/mock_vllm_server.py" > /tmp/mock_vllm_server.log 2>&1 &
SERVER_PID=$!

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping mock server (PID: $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    echo "Cleanup complete"
}
trap cleanup EXIT

# Wait for server to start
echo "Waiting for server to be ready..."
for i in {1..20}; do
    if curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Server is ready${NC}"
        break
    fi
    if [ $i -eq 20 ]; then
        echo -e "${RED}✗ Server failed to start${NC}"
        echo "Server log:"
        cat /tmp/mock_vllm_server.log
        exit 1
    fi
    sleep 0.5
done

# Run tests
echo ""
echo "Running tests..."
echo "=========================================="
python "$SCRIPT_DIR/test_text_completion.py"
TEST_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Deploy VLLM server on GPU machine"
    echo "2. Update base_url to point to GPU server"
    echo "3. Use the same code for production"
else
    echo -e "${RED}✗ Tests failed${NC}"
    echo ""
    echo "Check server log:"
    cat /tmp/mock_vllm_server.log
fi

exit $TEST_EXIT_CODE

#!/bin/bash
# Set environment variables for better performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYOPENGL_PLATFORM=egl

# Start the server
echo "🌟 Starting HaMeR API server on http://localhost:8000"
echo "📖 Visit http://localhost:8000/docs for interactive API documentation"
echo "💡 Press Ctrl+C to stop the server"
echo ""

python hamer_api.py
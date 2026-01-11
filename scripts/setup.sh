#!/bin/bash

# Setup script for the chatbot project

echo "🤖 Setting up General Chatbot project..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ $NODE_VERSION -lt 18 ]; then
    echo "❌ Node.js version 18+ is required. Current version: $(node -v)"
    exit 1
fi

echo "✅ Node.js version: $(node -v)"

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Copy environment file
if [ ! -f .env ]; then
    echo "📝 Creating environment file..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your actual configuration values"
fi

# Create required directories if they don't exist
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p data/{raw,processed,embeddings}

# Build the project
echo "🏗️  Building project..."
npm run build

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Set up your database (MongoDB/Redis)"
echo "3. Configure your LLM API keys"
echo "4. Run 'npm run dev' to start development server"

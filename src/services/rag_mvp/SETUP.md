# Setup Guide

Complete installation and configuration instructions for the RAG MVP.

---

## Prerequisites

### System Requirements
- **OS:** macOS (tested on M3 MacBook Air)
- **RAM:** 8GB minimum, 16GB+ recommended
- **Python:** 3.9 or 3.10 (avoid 3.12+)
- **Disk:** 1GB minimum

### Accounts Needed
- OpenAI API account with payment method
- Budget: $5-10 for testing

---

## Installation Steps

### Step 1: Create Project Directory

```bash
mkdir rag-mvp
cd rag-mvp
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# You should see (venv) in your terminal prompt
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install packages in this exact order
pip install openai==1.3.7
pip install "numpy<2.0"
pip install chromadb==0.4.22
pip install pypdf==3.17.4
pip install python-dotenv==1.0.0
```

**Why these specific versions?**
- `openai==1.3.7` - Stable, compatible with Python 3.9
- `numpy<2.0` - ChromaDB requires numpy 1.x
- Other packages have compatible dependencies

### Step 4: Verify Installation

```bash
python -c "import openai; import chromadb; print('✓ All packages installed successfully!')"
```

**Expected output:** `✓ All packages installed successfully!`

---

## OpenAI API Setup

### Step 1: Create Account

1. Go to https://platform.openai.com/signup
2. Sign up with your email
3. Verify your email address

### Step 2: Add Payment Method

1. Go to https://platform.openai.com/account/billing/overview
2. Click "Add payment method"
3. Add your credit card
4. Set usage limit to $10 (recommended for safety)

### Step 3: Create API Key

1. Go to https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Name it "RAG-MVP"
4. **Copy the key** (you won't see it again!)
5. Key format: `sk-proj-...` (long string)

---

## Configuration

### Step 1: Create .env File

```bash
# Create .env file
touch .env

# Open with text editor
nano .env
```

Add this line (paste your actual key):
```
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

Save and exit:
- Press `Ctrl+X`
- Press `Y` to confirm
- Press `Enter`

### Step 2: Create .gitignore

```bash
cat > .gitignore << EOF
.env
venv/
*.pyc
__pycache__/
chroma_db/
.DS_Store
EOF
```

**Important:** This prevents committing your API key to git!

### Step 3: Verify API Key Loading

```bash
python test_env.py
```

**Expected output:**
```
API Key loaded: sk-proj-TA...
Key length: 164
```

---

## Project Files Setup

### Step 1: Create Documents Folder

```bash
mkdir documents
```

### Step 2: Add PDF Files

Add 5-10 PDF files to the `documents/` folder.

**Don't have PDFs?** Download from:
- Research papers: https://arxiv.org/
- Free books: https://www.gutenberg.org/
- Technical docs: Any PDF documentation

**Verify:**
```bash
ls documents/
# Should show your PDF files
```

---

## Verification Checklist

Run through this checklist to ensure everything is set up correctly:

- [ ] Virtual environment created and activated
- [ ] All packages installed without errors
- [ ] OpenAI API key obtained
- [ ] `.env` file created with API key
- [ ] `.gitignore` file created
- [ ] `test_env.py` shows API key loaded
- [ ] Documents folder contains 5-10 PDFs

---

## Common Setup Issues

### Issue: Python version too old/new

```bash
# Check Python version
python3 --version

# Should be 3.9.x or 3.10.x
# If not, install correct version via Homebrew:
brew install python@3.10
python3.10 -m venv venv
```

### Issue: Package installation fails

```bash
# Clean reinstall
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
# Install packages one by one
```

### Issue: API key not loading

```bash
# Check .env file format
cat .env

# Must be exactly:
# OPENAI_API_KEY=sk-proj-...
# No spaces around =
# No quotes around key
```

---

## Next Steps

Once setup is complete:

1. **Test the pipeline:** See [USAGE.md](USAGE.md)
2. **Understand how it works:** See [HOW_IT_WORKS.md](HOW_IT_WORKS.md)
3. **Run full tests:** See [TESTING.md](TESTING.md)

---

**Setup complete?** → [USAGE.md](USAGE.md)

**Having issues?** → [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

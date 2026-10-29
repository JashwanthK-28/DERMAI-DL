#!/usr/bin/env bash
# DermAI: Skin Disease Detector from Mobile Images Using Deep Learning
# Infosys Springboard Internship - Startup Script
# ===================================================
# Usage: bash start.sh

set -e

VENV="./venv"
BACKEND_PORT=8000
FRONTEND_PORT=5173

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║       DermAI: Skin Disease Detector using Deep Learning          ║"
echo "║       Infosys Springboard AI Internship — Startup Script         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Activate virtual environment ──────────────────────────────────────────
if [ ! -d "$VENV" ]; then
  echo "[!] Virtual environment not found. Creating at $VENV..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install -r backend/requirements.txt
  "$VENV/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# ── 2. Check required model checkpoints ──────────────────────────────────────
if [ ! -f "saved_models/resnet50_baseline.pth" ]; then
  echo ""
  echo "[!] ResNet50 checkpoint not found. Training baseline model (2 epochs, CPU-fast)..."
  "$VENV/bin/python" train_baseline.py --epochs 3 --batch_size 16 --subset 400
fi

if [ ! -f "saved_models/efficientnet_b0.pth" ]; then
  echo ""
  echo "[!] EfficientNet-B0 checkpoint not found. Training model (2 epochs, CPU-fast)..."
  "$VENV/bin/python" train_efficientnet.py --epochs 3 --batch_size 16 --subset 400
fi

# ── 3. Generate segmentation demo artifacts ───────────────────────────────────
if [ ! -d "outputs/segmentation" ] || [ -z "$(ls -A outputs/segmentation 2>/dev/null)" ]; then
  echo ""
  echo "[*] Generating Task 3 lesion segmentation prototype visual artifacts..."
  "$VENV/bin/python" lesion_segmentation.py
fi

# ── 4. Run model comparison evaluation ───────────────────────────────────────
if [ ! -f "outputs/model_comparison_metrics.json" ]; then
  echo ""
  echo "[*] Running Task 2 Dual CNN comparison evaluation pipeline..."
  "$VENV/bin/python" evaluate_models.py
fi

# ── 5. Run pytest validation suite ───────────────────────────────────────────
echo ""
echo "[*] Running Section 5 Validation Checks (4 pytest tests)..."
"$VENV/bin/python" -m pytest tests/test_all.py -v --tb=short 2>&1 || {
  echo "[!] Some tests failed — see output above. Backend can still start."
}

# ── 6. Start FastAPI Backend ──────────────────────────────────────────────────
echo ""
echo "[*] Starting FastAPI Backend on http://localhost:$BACKEND_PORT ..."
echo "    API Docs available at: http://localhost:$BACKEND_PORT/docs"
"$VENV/bin/uvicorn" backend.app.main:app --host 0.0.0.0 --port $BACKEND_PORT --reload &
BACKEND_PID=$!
echo "    Backend PID: $BACKEND_PID"

# Wait for backend to initialize
sleep 2

# ── 7. Start React Frontend Dev Server ───────────────────────────────────────
echo ""
echo "[*] Starting React Frontend Dev Server on http://localhost:$FRONTEND_PORT ..."
cd frontend
npm run dev &
FRONTEND_PID=$!
echo "    Frontend PID: $FRONTEND_PID"
cd ..

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                  DermAI Platform is LIVE!                        ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Frontend App:    http://localhost:5173                          ║"
echo "║  Backend API:     http://localhost:8000                          ║"
echo "║  Swagger Docs:    http://localhost:8000/docs                     ║"
echo "║  ReDoc API Docs:  http://localhost:8000/redoc                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Press Ctrl+C to stop all services."
echo ""

# Keep alive until user stops
wait

#!/bin/bash

# 個別ファイル版最適化テスト実行スクリプト
# 各最適化技法を個別のファイルで実行し、性能を比較する

set -e

echo "=== 個別ファイル版最適化テスト実行スクリプト ==="
echo "開始時刻: $(date)"
echo ""

# ログファイル名
LOG_FILE="separate_optimization_results_$(date +%Y%m%d_%H%M%S).log"

# ログファイルに出力を保存
exec > >(tee "$LOG_FILE") 2>&1

echo "ログファイル: $LOG_FILE"
echo ""

# 環境確認
echo "=== 環境確認 ==="
echo "CPU情報:"
lscpu | grep -E "(Model name|CPU\(s\)|Thread|Core|Socket)"
echo ""
echo "コンパイラ確認:"
which icc && icc --version | head -1
echo ""
echo "OpenMP設定:"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-未設定}"
echo ""

# コンパイル
echo "=== コンパイル ==="
make -f Makefile-separate clean
make -f Makefile-separate all
echo "コンパイル完了"
echo ""

# テスト実行
echo "=== テスト実行 ==="

# 基本版テスト
echo "--- 基本版テスト（最適化なし） ---"
make -f Makefile-separate run-basic
echo ""

# ブロッキング版テスト
echo "--- ブロッキング版テスト ---"
make -f Makefile-separate run-blocking
echo ""

# AVX2版テスト
echo "--- AVX2版テスト ---"
make -f Makefile-separate run-avx2
echo ""

# OpenMP版テスト
echo "--- OpenMP版テスト ---"
make -f Makefile-separate run-openmp
echo ""

# MKL版テスト
echo "--- MKL版テスト ---"
make -f Makefile-separate run-mkl
echo ""

# ループアンローリング版テスト
echo "--- ループアンローリング版テスト ---"
make -f Makefile-separate run-unroll
echo ""

echo "=== テスト完了 ==="
echo "終了時刻: $(date)"
echo ""

# 結果分析
echo "=== 結果分析 ==="
echo "ログファイルからGFLOPSを抽出して分析します..."
echo ""

# 各最適化技法の最高GFLOPSを抽出
echo "各最適化技法の最高性能:"
echo "基本版: $(grep 'GFLOPS' "$LOG_FILE" | grep 'unoptimized' | awk '{print $4}' | sort -nr | head -1) GFLOPS"
echo "loop exchange: $(grep 'GFLOPS' "$LOG_FILE" | grep 'loop exchange' | awk '{print $4}' | sort -nr | head -1) GFLOPS"
echo "blocking: $(grep 'GFLOPS' "$LOG_FILE" | grep 'blocking' | awk '{print $4}' | sort -nr | head -1) GFLOPS"
echo "AVX2: $(grep 'GFLOPS' "$LOG_FILE" | grep 'AVX2' | awk '{print $4}' | sort -nr | head -1) GFLOPS"
echo "OpenMP: $(grep 'GFLOPS' "$LOG_FILE" | grep 'OpenMP' | awk '{print $4}' | sort -nr | head -1) GFLOPS"
echo "MKL: $(grep 'GFLOPS' "$LOG_FILE" | grep 'MKL' | awk '{print $4}' | sort -nr | head -1) GFLOPS"
echo "Loop Unrolling: $(grep 'GFLOPS' "$LOG_FILE" | grep 'Loop Unrolling' | awk '{print $4}' | sort -nr | head -1) GFLOPS"
echo ""

# 全体的な最高性能
echo "全体的な最高性能:"
grep 'GFLOPS' "$LOG_FILE" | awk '{print $4, $6}' | sort -nr | head -5
echo ""

echo "=== 分析完了 ==="
echo "詳細な結果はログファイル $LOG_FILE を確認してください。"

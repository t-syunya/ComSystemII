#!/bin/bash

# 行列行列積最適化課題実行スクリプト
# このスクリプトは手順書に従って最適化テストを自動実行します

set -e  # エラー時に停止

# 色付き出力のための関数
print_header() {
    echo -e "\n\033[1;34m=== $1 ===\033[0m"
}

print_success() {
    echo -e "\033[1;32m✓ $1\033[0m"
}

print_info() {
    echo -e "\033[1;36mℹ $1\033[0m"
}

print_warning() {
    echo -e "\033[1;33m⚠ $1\033[0m"
}

print_error() {
    echo -e "\033[1;31m✗ $1\033[0m"
}

# 結果保存用ファイル
RESULTS_FILE="optimization_results_$(date +%Y%m%d_%H%M%S).txt"
LOG_FILE="optimization_log_$(date +%Y%m%d_%H%M%S).txt"

# ログ関数
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# 初期化
init() {
    print_header "行列行列積最適化課題実行スクリプト"
    log "スクリプト開始"
    
    # 結果ファイルの初期化
    echo "=== 行列行列積最適化性能テスト ===" > "$RESULTS_FILE"
    echo "実行日時: $(date)" >> "$RESULTS_FILE"
    echo "実行環境: $(uname -a)" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    
    # CPU情報の記録
    echo "CPU情報:" >> "$RESULTS_FILE"
    lscpu | grep "Model name\|CPU(s)\|Thread\|Core" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    
    # コンパイラ情報の記録
    echo "コンパイラ情報:" >> "$RESULTS_FILE"
    which gcc >> "$RESULTS_FILE" 2>&1 || echo "gcc: not found" >> "$RESULTS_FILE"
    which icc >> "$RESULTS_FILE" 2>&1 || echo "icc: not found" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
}

# 環境確認
check_environment() {
    print_header "環境確認"
    
    # CPU情報
    print_info "CPU情報:"
    lscpu | grep "Model name\|CPU(s)\|Thread\|Core"
    
    # コンパイラ確認
    print_info "コンパイラ確認:"
    if command -v gcc &> /dev/null; then
        print_success "GCC: $(gcc --version | head -1)"
    else
        print_warning "GCC: not found"
    fi
    
    if command -v icc &> /dev/null; then
        print_success "ICC: $(icc --version | head -1)"
    else
        print_warning "ICC: not found"
    fi
    
    # OpenMP確認
    print_info "OpenMP確認:"
    echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-not set}"
    
    log "環境確認完了"
}

# 基本テスト
run_basic_test() {
    print_header "基本テスト（最適化なし）"
    
    echo "1. 基本テスト（最適化なし）" >> "$RESULTS_FILE"
    
    make clean
    make
    
    print_info "基本テスト実行中..."
    make run | tee -a "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    
    print_success "基本テスト完了"
    log "基本テスト完了"
}

# ブロッキングテスト
run_blocking_tests() {
    print_header "ブロッキングテスト"
    
    local block_sizes=(16 32 64 128)
    
    for block_size in "${block_sizes[@]}"; do
        print_info "ブロックサイズ: $block_size"
        
        echo "2. ブロッキングテスト（ブロックサイズ: $block_size）" >> "$RESULTS_FILE"
        
        # BLOCKSIZEを変更
        sed -i "s/#define BLOCKSIZE [0-9]*/#define BLOCKSIZE $block_size/" gemm-test.c
        sed -i 's|//#define BLOCKING|#define BLOCKING|' gemm-test.c
        
        make clean
        make
        
        print_info "ブロッキングテスト実行中（ブロックサイズ: $block_size）..."
        make run | tee -a "$RESULTS_FILE"
        echo "" >> "$RESULTS_FILE"
        
        # 元に戻す
        sed -i 's|#define BLOCKING|//#define BLOCKING|' gemm-test.c
        
        print_success "ブロックサイズ $block_size のテスト完了"
    done
    
    log "ブロッキングテスト完了"
}

# AVX2テスト
run_avx2_test() {
    print_header "AVX2ベクトル化テスト"
    
    echo "3. AVX2テスト" >> "$RESULTS_FILE"
    
    sed -i 's|//#define AVX2|#define AVX2|' gemm-test.c
    make clean
    make
    
    print_info "AVX2テスト実行中..."
    make run | tee -a "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    
    # 元に戻す
    sed -i 's|#define AVX2|//#define AVX2|' gemm-test.c
    
    print_success "AVX2テスト完了"
    log "AVX2テスト完了"
}

# OpenMPテスト
run_openmp_tests() {
    print_header "OpenMP並列化テスト"
    
    local thread_counts=(1 2 4 8)
    
    echo "4. OpenMPテスト" >> "$RESULTS_FILE"
    
    sed -i 's|//#define OMP|#define OMP|' gemm-test.c
    make clean
    make
    
    for threads in "${thread_counts[@]}"; do
        print_info "スレッド数: $threads"
        
        echo "スレッド数: $threads" >> "$RESULTS_FILE"
        export OMP_NUM_THREADS=$threads
        
        print_info "OpenMPテスト実行中（スレッド数: $threads）..."
        make run | tee -a "$RESULTS_FILE"
        echo "" >> "$RESULTS_FILE"
        
        print_success "スレッド数 $threads のテスト完了"
    done
    
    # 元に戻す
    sed -i 's|#define OMP|//#define OMP|' gemm-test.c
    
    log "OpenMPテスト完了"
}

# AVX2 + OpenMPテスト
run_avx2_openmp_tests() {
    print_header "AVX2 + OpenMP組み合わせテスト"
    
    local thread_counts=(1 2 4 8)
    
    echo "5. AVX2 + OpenMPテスト" >> "$RESULTS_FILE"
    
    sed -i 's|//#define AVX_OMP|#define AVX_OMP|' gemm-test.c
    make clean
    make
    
    for threads in "${thread_counts[@]}"; do
        print_info "AVX2 + OpenMP スレッド数: $threads"
        
        echo "AVX2 + OpenMP スレッド数: $threads" >> "$RESULTS_FILE"
        export OMP_NUM_THREADS=$threads
        
        print_info "AVX2 + OpenMPテスト実行中（スレッド数: $threads）..."
        make run | tee -a "$RESULTS_FILE"
        echo "" >> "$RESULTS_FILE"
        
        print_success "AVX2 + OpenMP スレッド数 $threads のテスト完了"
    done
    
    # 元に戻す
    sed -i 's|#define AVX_OMP|//#define AVX_OMP|' gemm-test.c
    
    log "AVX2 + OpenMPテスト完了"
}

# MKLテスト
run_mkl_test() {
    print_header "MKLライブラリテスト"
    
    echo "6. MKLテスト" >> "$RESULTS_FILE"
    
    sed -i 's|//#define MKL|#define MKL|' gemm-test.c
    make clean
    make
    
    print_info "MKLテスト実行中..."
    make run | tee -a "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    
    # 元に戻す
    sed -i 's|#define MKL|//#define MKL|' gemm-test.c
    
    print_success "MKLテスト完了"
    log "MKLテスト完了"
}

# ループアンローリングテスト（実装が必要）
run_unroll_test() {
    print_header "ループアンローリングテスト"
    
    print_warning "ループアンローリングの実装が必要です"
    print_info "gemm-test.cにdgemm_unroll関数を追加してください"
    print_info "また、UNROLL_ONLYのdefineを有効化してください"
    
    echo "7. ループアンローリングテスト（要実装）" >> "$RESULTS_FILE"
    echo "実装が必要です" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    
    log "ループアンローリングテスト（要実装）"
}

# 結果分析
analyze_results() {
    print_header "結果分析"
    
    echo "" >> "$RESULTS_FILE"
    echo "=== 結果サマリー ===" >> "$RESULTS_FILE"
    echo "最高性能の組み合わせ（上位10位）:" >> "$RESULTS_FILE"
    grep "GFLOPS" "$RESULTS_FILE" | sort -k3 -nr | head -10 >> "$RESULTS_FILE"
    
    echo "" >> "$RESULTS_FILE"
    echo "=== 最適化効果の分析 ===" >> "$RESULTS_FILE"
    
    # 基本性能
    local baseline=$(grep "unoptimized" "$RESULTS_FILE" | tail -1 | awk '{print $4}' 2>/dev/null || echo "0")
    echo "1. 基本性能（最適化なし）: $baseline GFLOPS" >> "$RESULTS_FILE"
    
    # 最高性能
    local best=$(grep "GFLOPS" "$RESULTS_FILE" | sort -k3 -nr | head -1 | awk '{print $4}' 2>/dev/null || echo "0")
    echo "2. 最高性能: $best GFLOPS" >> "$RESULTS_FILE"
    
    # 性能向上率
    if [ "$baseline" != "0" ] && [ "$best" != "0" ]; then
        local improvement=$(echo "scale=2; $best / $baseline" | bc 2>/dev/null || echo "N/A")
        echo "3. 性能向上率: ${improvement}x" >> "$RESULTS_FILE"
    else
        echo "3. 性能向上率: 計算できません" >> "$RESULTS_FILE"
    fi
    
    # 結果表示
    print_info "結果ファイル: $RESULTS_FILE"
    print_info "ログファイル: $LOG_FILE"
    
    echo ""
    print_header "最高性能の組み合わせ（上位5位）"
    grep "GFLOPS" "$RESULTS_FILE" | sort -k3 -nr | head -5
    
    if [ "$baseline" != "0" ] && [ "$best" != "0" ]; then
        local improvement=$(echo "scale=2; $best / $baseline" | bc 2>/dev/null || echo "N/A")
        echo ""
        print_success "性能向上率: ${improvement}x"
    fi
    
    log "結果分析完了"
}

# クリーンアップ
cleanup() {
    print_header "クリーンアップ"
    
    # 元の状態に戻す
    sed -i 's|#define BLOCKING|//#define BLOCKING|' gemm-test.c 2>/dev/null || true
    sed -i 's|#define AVX2|//#define AVX2|' gemm-test.c 2>/dev/null || true
    sed -i 's|#define OMP|//#define OMP|' gemm-test.c 2>/dev/null || true
    sed -i 's|#define AVX_OMP|//#define AVX_OMP|' gemm-test.c 2>/dev/null || true
    sed -i 's|#define MKL|//#define MKL|' gemm-test.c 2>/dev/null || true
    sed -i 's/#define BLOCKSIZE [0-9]*/#define BLOCKSIZE 32/' gemm-test.c 2>/dev/null || true
    
    make clean
    
    print_success "クリーンアップ完了"
    log "クリーンアップ完了"
}

# メイン実行関数
main() {
    init
    check_environment
    
    # 各テストの実行
    run_basic_test
    run_blocking_tests
    run_avx2_test
    run_openmp_tests
    run_avx2_openmp_tests
    run_mkl_test
    run_unroll_test
    
    # 結果分析
    analyze_results
    
    # クリーンアップ
    cleanup
    
    print_header "テスト完了"
    print_success "全てのテストが完了しました"
    print_info "結果ファイル: $RESULTS_FILE"
    print_info "ログファイル: $LOG_FILE"
    
    log "スクリプト完了"
}

# ヘルプ表示
show_help() {
    echo "使用方法: $0 [オプション]"
    echo ""
    echo "オプション:"
    echo "  -h, --help     このヘルプを表示"
    echo "  -v, --verbose  詳細なログを出力"
    echo ""
    echo "例:"
    echo "  $0             通常実行"
    echo "  $0 -v          詳細ログ付きで実行"
}

# オプション解析
VERBOSE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            print_error "不明なオプション: $1"
            show_help
            exit 1
            ;;
    esac
done

# メイン実行
main "$@"

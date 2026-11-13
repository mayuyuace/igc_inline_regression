#!/usr/bin/env bash
# Offline reproducer for the IGC inlining regression (no GPU / PyTorch needed).
#
# Compiles topk_per_row.spv with the installed ocloc/IGC and reports, for
# kernel _ZTSN4vllm28top_k_per_row_prefill_kernelILi512ELb0EEE:
#   - number of calls left to the loadChunk lambda after IGC optimization
#   - private_size from zeinfo
#   - load.ugm / store.ugm counts in the final ISA
#
# Good (IGC 26.43.40581): calls=0, private_size=0,    load.ugm~240 store.ugm~59
# Bad  (IGC 26.44.40863): calls=80, private_size=5888, load.ugm~463 store.ugm~186
#
# Usage: ./repro.sh [device_id]      (default 0x674f)
set -euo pipefail
cd "$(dirname "$0")"
DEV=${1:-0x674f}
KERNEL=_ZTSN4vllm28top_k_per_row_prefill_kernelILi512ELb0EEE
LAMBDA='topKPerRowReduce.*clEiSC_SE_'
OUT=$(mktemp -d /tmp/igc_inline_XXXX)

echo "ocloc     : $(command -v ocloc)"
ocloc --version 2>/dev/null | head -2 || true
dpkg -l 2>/dev/null | awk '/intel-opencl-icd |libigc2 |libigc1 /{print "pkg       : "$2" "$3}' || true
echo "device    : $DEV"
echo "dump dir  : $OUT"

IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir="$OUT" \
  ocloc compile -spirv_input -file topk_per_row.spv -device "$DEV" \
  -options '-cl-intel-greater-than-4GB-buffer-required' \
  -output_no_suffix -out_dir "$OUT/bin" >/dev/null

LL=$(ls "$OUT"/*_optimized.ll | grep -v '_1_' | head -1)
ZE=$(ls "$OUT"/*.zeinfo | head -1)
ASM=$(grep -l "$KERNEL" "$OUT"/*simd*_entry_*.asm | head -1)

calls=$(grep -cE "call .*${LAMBDA}" "$LL" || true)
priv=$(awk -v k="$KERNEL" '$0 ~ "name:.*"k {f=1; next} f && /- name:/ {f=0} f && /private_size/ {print $2; exit}' "$ZE")
ld=$(grep -oE 'load\.ugm' "$ASM" | wc -l)
st=$(grep -oE 'store\.ugm' "$ASM" | wc -l)

echo
echo "kernel                         : $KERNEL"
echo "non-inlined loadChunk calls    : $calls"
echo "private_size (bytes / WI)      : ${priv:-0}"
echo "load.ugm / store.ugm in ISA    : $ld / $st"
echo
if [ "${calls:-0}" -gt 0 ] || [ "${priv:-0}" -gt 0 ]; then
  echo "RESULT: BAD  - lambda not inlined, by-ref arrays demoted to private memory"
  exit 1
fi
echo "RESULT: GOOD - lambda inlined, no private memory"

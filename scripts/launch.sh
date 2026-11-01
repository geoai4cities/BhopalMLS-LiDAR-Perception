#!/usr/bin/env bash
################################################################################
# BhopalMLS benchmark launcher. The per-model wrappers (<Model>/Bhopal/scripts/*.sh,
# <Model>/<Dataset>/scripts/{train,test}.sh, <Model>/cross_data_testing/run_cross_test.sh) all call this.
#
#   scripts/launch.sh train <model> <protocol> [tokens] [options] [key=value ...]
#   scripts/launch.sh test  <model> <protocol> [tokens] [options] [--weight PATH]
#   scripts/launch.sh cross <model> <source>_on_<target> --weight PATH
#
#   model     ptv3 | litept | kpconvx | sonata | concerto | utonia
#   protocol  supervised:  train | insseg          foundation:  lp | dp | lora | ft | insseg_lp | insseg_dp | insseg_ft
#   tokens    4ch | 9ch (input; default 9ch)      12c | 21c (taxonomy; default 12c)      v10 (68-class file, dagger rows)
#             outdoor | indoor (Concerto checkpoint: outdoor = concerto_large_outdoor.pth = Concerto(O), the default;
#             indoor = concerto_large.pth = Concerto(L); run folders get the _indoor suffix as in the paper's runs)
#   options   --dataset D     Bhopal (default) | Kerala | Toronto-3D | Paris-Lille-3D | SemanticKITTI: the folder whose
#                             configs/ and outputs/ are used (the dataset wrappers set it; Bhopal only for v10 and 21c)
#             --config PATH   any config (ablations, ...); the output folder follows the config's <Model>/<Dataset>/ folder
#             --name NAME     output folder name            --gpu N          CUDA device (default 0)
#             --batch B --accum A   override the paper's batch size / gradient accumulation
#             --resume        continue from <run>/model/model_last.pth          --dry-run   print the command only
#
# Paths are relative to the repository root, which the launcher cds into (a relative --config or --weight that exists
# relative to where the command was typed is accepted as well): data/<Dataset>/*.h5, <Model>/<Dataset>/configs/...,
# outputs in <Model>/<Dataset>/outputs/<run>/, cross-region outputs in <Model>/cross_data_testing/outputs/<pair>/ft_9ch/.
# Paper batch settings (effective batch = batch x accumulation), as run on a 143 GB H200:
#   semantic: supervised 12 x 1; foundation LP 32 x 2, DP 16 x 4 (sonata 32 x 2), LoRA 8 x 8 (sonata 32 x 2), FT 8 x 8 (sonata 32 x 2)
#   instance: supervised 12 x 1; foundation LP/DP 4 x 6, FT 2 x 12 (utonia) or 8 x 3 (sonata, concerto)
#   On a 40 GB card halve the batch and double the accumulation; the effective batch is what matters.
################################################################################
set -euo pipefail
CALLER="$PWD"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
if [[ -z "${VIRTUAL_ENV:-}" && -f bhopalmls_venv/bin/activate ]]; then source ./activate.sh >/dev/null; fi
usage() { awk 'NR > 1 && /^set -euo/ { exit } NR > 1 { sub(/^#+ ?/, ""); print }' "$0"; exit "${1:-0}"; }
resolve() {  # a path as typed -> a path that works from the repository root
  local p=$1
  if [[ "$p" == /* || -e "$p" ]]; then echo "$p"; elif [[ -e "$CALLER/$p" ]]; then echo "$CALLER/$p"; else echo "$p"; fi
}
arr() { local n=$1; eval "echo \${#$n[@]}"; }

[[ $# -ge 3 ]] || usage 1
MODE=$1; MODEL=$2; shift 2
case "$MODE" in train|test|cross) ;; -h|--help) usage;; *) echo "mode must be train, test or cross"; usage 1;; esac
case "$MODEL" in ptv3|litept|kpconvx|sonata|concerto|utonia) ;; *) echo "unknown model: $MODEL"; usage 1;; esac
ARCH_DIR=$(case "$MODEL" in ptv3) echo PTv3;; litept) echo LitePT;; kpconvx) echo KpConvX;; sonata) echo Sonata;; concerto) echo Concerto;; utonia) echo Utonia;; esac)
FM=0; case "$MODEL" in sonata|concerto|utonia) FM=1;; esac
if [[ "$MODEL" == litept ]]; then FW="LitePT/litept"; export PYTHONPATH="$REPO/LitePT/litept/libs/pointrope:$REPO/$FW:${PYTHONPATH:-}"
else FW="Pointcept"; export PYTHONPATH="$REPO/$FW:${PYTHONPATH:-}"; fi

# ---------------------------------------------------------------- cross-region evaluation
if [[ "$MODE" == cross ]]; then
  PAIR=$1; shift; WEIGHT=""; GPU=0; DRY=0; EXTRA=()
  while [[ $# -gt 0 ]]; do case "$1" in --weight) WEIGHT=$(resolve "$2"); shift 2;; --gpu) GPU=$2; shift 2;; --dry-run) DRY=1; shift;; *=*) EXTRA+=("$1"); shift;; *) echo "unknown argument $1"; exit 1;; esac; done
  CFG="$ARCH_DIR/cross_data_testing/configs/${MODEL}_crosscity_${PAIR}_9ch.py"
  [[ -f "$CFG" ]] || { echo "no transfer config for $MODEL $PAIR ($CFG)"; exit 1; }
  OUT="$ARCH_DIR/cross_data_testing/outputs/$PAIR/ft_9ch"
  CMD=(python "$FW/tools/test.py" --config-file "$CFG" --options "save_path=$OUT" batch_size_test=1 num_worker=8 "weight=$WEIGHT" ${EXTRA[@]+"${EXTRA[@]}"})
  echo "config  : $CFG"; echo "weight  : $WEIGHT"; echo "output  : $OUT"; echo "command : CUDA_VISIBLE_DEVICES=$GPU ${CMD[*]}"
  [[ "$DRY" == 1 ]] && exit 0
  [[ -n "$WEIGHT" && -f "$WEIGHT" ]] || { echo "--weight PATH (existing file) is required"; exit 1; }
  mkdir -p "$OUT/logs"; [[ -d "$OUT/result" ]] && find "$OUT/result" -name '*_pred.npy' -delete
  export CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1
  "${CMD[@]}" 2>&1 | tee "$OUT/logs/test_$(date +%Y%m%d_%H%M%S).log"
  exit
fi

# ---------------------------------------------------------------- train / test: parse
PROTO=$1; shift
CH=9ch; NC=12; MOVING=0; PRETRAIN=outdoor; DS=Bhopal; CONFIG=""; NAME=""; GPU=0; BATCH=""; ACCUM=""; RESUME=0; DRY=0; WEIGHT=""; EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    4ch|9ch) CH=$1; shift;; 12c) NC=12; shift;; 21c) NC=21; shift;; v10) MOVING=1; shift;;
    outdoor) PRETRAIN=outdoor; shift;; indoor|large) PRETRAIN=indoor; shift;;
    --dataset) DS=$2; shift 2;;
    --channels) CH=$2; shift 2;; --classes) NC=$2; shift 2;; --moving) MOVING=1; shift;; --pretrain) PRETRAIN=$2; shift 2;;
    --config) CONFIG=$(resolve "$2"); shift 2;; --name) NAME=$2; shift 2;; --gpu) GPU=$2; shift 2;;
    --batch) BATCH=$2; shift 2;; --accum) ACCUM=$2; shift 2;; --resume) RESUME=1; shift;; --dry-run) DRY=1; shift;;
    --weight) WEIGHT=$(resolve "$2"); shift 2;; -h|--help) usage;;
    *=*) EXTRA+=("$1"); shift;; *) echo "unknown argument: $1"; usage 1;;
  esac
done
[[ "$CH" == 4ch || "$CH" == 9ch ]] || { echo "channels must be 4ch or 9ch"; exit 1; }
[[ "$NC" == 12 || "$NC" == 21 ]] || { echo "classes must be 12c or 21c"; exit 1; }
case "$PRETRAIN" in outdoor|indoor) ;; large) PRETRAIN=indoor;; *) echo "pretrain must be outdoor or indoor"; exit 1;; esac
# dataset folder -> config sub-folder and the dataset tag of the run names (the ones of the paper's runs)
case "$(echo "$DS" | tr 'A-Z' 'a-z')" in
  bhopal) DS=Bhopal; SUB=bhopal; DT=bhopal;;
  kerala|ke-mls) DS=Kerala; SUB=kerala; DT=kerala;;
  toronto-3d|toronto) DS=Toronto-3D; SUB=toronto; DT=toronto;;
  paris-lille-3d|paris_lille|pl3d) DS=Paris-Lille-3D; SUB=paris_lille; DT=paris_lille;;
  semantickitti|kitti|sk) DS=SemanticKITTI; SUB=semantickitti; DT=kitti;;
  *) echo "unknown dataset '$DS' (Bhopal, Kerala, Toronto-3D, Paris-Lille-3D, SemanticKITTI)"; exit 1;;
esac
if [[ "$DS" != Bhopal && "$NC" == 21 ]]; then echo "the 21-class taxonomy exists for BhopalMLS only"; exit 1; fi
NCTAG="${NC}class"; SFX=""; [[ "$NC" == 21 ]] && SFX="_21c"
CDIR="$ARCH_DIR/$DS/configs/$SUB"

# ---------------------------------------------------------------- config + paper batch settings
case "$MODEL:$PROTO" in
  ptv3:train)      CFG="$CDIR/ptv3_140gb_${NCTAG}_train_${CH}.py";      B=12; A=1;  RUN="ptv3_${DT}_train_${CH}${SFX}";;
  ptv3:insseg)     CFG="$CDIR/ptv3_140gb_${NCTAG}_insseg_9ch.py";       B=12; A=1;  RUN="ptv3_${DT}_insseg_9ch${SFX}";;
  kpconvx:train)   CFG="$CDIR/kpconvx_140gb_${NCTAG}_train_${CH}.py";   B=12; A=1;  RUN="kpconvx_${DT}_train_${CH}${SFX}";;
  litept:train)    CFG="$CDIR/litept_L_${NCTAG}_${CH}.py";              B=6;  A=2;  RUN="litept_L_${DT}_${CH}${SFX}";;
  litept:insseg)   CFG="$CDIR/litept_L_${NCTAG}_insseg_9ch.py";         B=12; A=1;  RUN="litept_L_${DT}_insseg_9ch${SFX}";;
  sonata:lp|concerto:lp|utonia:lp)                   CFG="$CDIR/${MODEL}_140gb_${NCTAG}_linear_probe_${CH}.py";  B=32; A=2;  RUN="${MODEL}_${DT}_linear_probe_${CH}${SFX}";;
  sonata:dp|concerto:dp|utonia:dp)                   CFG="$CDIR/${MODEL}_140gb_${NCTAG}_decoder_probe_${CH}.py"; B=16; A=4;  RUN="${MODEL}_${DT}_decoder_probe_${CH}${SFX}";;
  sonata:lora|concerto:lora|utonia:lora)             CFG="$CDIR/${MODEL}_140gb_${NCTAG}_lora_${CH}.py";          B=8;  A=8;  RUN="${MODEL}_${DT}_lora_${CH}${SFX}";;
  sonata:ft|concerto:ft|utonia:ft)                   CFG="$CDIR/${MODEL}_140gb_${NCTAG}_finetune_${CH}.py";      B=8;  A=8;  RUN="${MODEL}_${DT}_finetune_${CH}${SFX}";;
  sonata:insseg_lp|concerto:insseg_lp|utonia:insseg_lp) CFG="$CDIR/${MODEL}_140gb_${NCTAG}_insseg_lp_${CH}.py";  B=4;  A=6;  RUN="${MODEL}_${DT}_insseg_lp_${CH}${SFX}";;
  sonata:insseg_dp|concerto:insseg_dp|utonia:insseg_dp) CFG="$CDIR/${MODEL}_140gb_${NCTAG}_insseg_dp_${CH}.py";  B=4;  A=6;  RUN="${MODEL}_${DT}_insseg_dp_${CH}${SFX}";;
  sonata:insseg_ft|concerto:insseg_ft|utonia:insseg_ft) CFG="$CDIR/${MODEL}_140gb_${NCTAG}_insseg_ft_${CH}.py";  B=2;  A=12; RUN="${MODEL}_${DT}_insseg_ft_${CH}${SFX}";;
  *) echo "unknown protocol '$PROTO' for $MODEL"; usage 1;;
esac
if [[ ! -f "$CFG" && -f "${CFG/_140gb_/_}" ]]; then CFG="${CFG/_140gb_/_}"; fi   # the SemanticKITTI configs carry no 140gb tag
[[ "$MODEL" == sonata ]] && case "$PROTO" in dp|lora|ft) B=32; A=2;; insseg_ft) B=8; A=3;; esac
[[ "$MODEL" == concerto && "$PROTO" == insseg_ft ]] && { B=8; A=3; }
if [[ "$MOVING" == 1 ]]; then
  [[ "$DS" == Bhopal ]] || { echo "v10 (the 68-class file with the moving classes) exists for BhopalMLS only"; exit 1; }
  [[ "$MODEL" == utonia || "$MODEL" == concerto ]] || { echo "v10 configs exist for utonia and concerto only"; exit 1; }
  case "$PROTO" in lp|dp|lora|ft) ;; *) echo "v10 configs exist for the semantic protocols lp, dp, lora and ft (the paper has no dagger instance rows)"; exit 1;; esac
  CFG="$ARCH_DIR/Bhopal/configs/bhopal_v10/$(basename "$CFG")"; RUN="${RUN}_v10"
fi
if [[ -n "$CONFIG" ]]; then
  CFG="$CONFIG"; RUN="$(basename "${CONFIG%.py}")"
  REL="${CONFIG#"$REPO"/}"                             # an explicit config decides the output folder: <Model>/<Dataset>/outputs/
  if [[ "$REL" =~ ^([^/]+)/([^/]+)/configs/ ]]; then ARCH_DIR="${BASH_REMATCH[1]}"; DS="${BASH_REMATCH[2]}"; fi
fi
if [[ "$MODEL" == concerto && "$PRETRAIN" == indoor && -z "$NAME" ]]; then RUN="${RUN}_indoor"; fi
[[ -n "$NAME" ]] && RUN="$NAME"
[[ -n "$BATCH" ]] && B="$BATCH"; [[ -n "$ACCUM" ]] && A="$ACCUM"
[[ -f "$CFG" ]] || { echo "config not found: $CFG"; exit 1; }
PRE=()
case "$MODEL" in
  utonia)   PRE=(weight=Utonia_pretrained/utonia.pth);;
  sonata)   PRE=(weight=sonata_pretrained/pretrain-sonata-v1m1-0-base.pth);;
  concerto) [[ "$PRETRAIN" == indoor ]] && PRE=(weight=Concerto_pretrained/concerto_large.pth) || PRE=(weight=Concerto_pretrained/concerto_large_outdoor.pth);;
esac
OUT="$ARCH_DIR/$DS/outputs/$RUN"

# ---------------------------------------------------------------- test
if [[ "$MODE" == test ]]; then
  [[ -n "$WEIGHT" ]] || WEIGHT="$OUT/model/model_best.pth"
  CMD=(python "$FW/tools/test.py" --config-file "$CFG" --options "save_path=$OUT" batch_size_test=1 num_worker=8 "weight=$WEIGHT" ${EXTRA[@]+"${EXTRA[@]}"})
  echo "run     : $RUN"; echo "config  : $CFG"; echo "weight  : $WEIGHT"; echo "output  : $OUT"; echo "command : CUDA_VISIBLE_DEVICES=$GPU ${CMD[*]}"
  [[ "$DRY" == 1 ]] && exit 0
  [[ -f "$WEIGHT" ]] || { echo "weight not found: $WEIGHT"; exit 1; }
  mkdir -p "$OUT/logs"; [[ -d "$OUT/result" ]] && find "$OUT/result" -name '*_pred.npy' -delete   # Pointcept reuses cached predictions silently
  export CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1
  "${CMD[@]}" 2>&1 | tee "$OUT/logs/test_$(date +%Y%m%d_%H%M%S).log"
  exit
fi

# ---------------------------------------------------------------- train
RES=()
if [[ "$RESUME" == 1 ]]; then
  CK="$OUT/model/model_last.pth"; [[ -f "$CK" ]] || { echo "nothing to resume at $CK"; exit 1; }
  RES=(resume=True "weight=$CK"); PRE=()
fi
CMD=(python "$FW/tools/train.py" --config-file "$CFG"
     --options "save_path=$OUT" "batch_size=$B" "gradient_accumulation_steps=$A" num_worker=16
     ${PRE[@]+"${PRE[@]}"} ${RES[@]+"${RES[@]}"} ${EXTRA[@]+"${EXTRA[@]}"})
echo "run     : $RUN"; echo "config  : $CFG"; echo "batch   : $B x $A (effective $((B*A)))"; echo "output  : $OUT"
echo "command : CUDA_VISIBLE_DEVICES=$GPU ${CMD[*]}"
[[ "$DRY" == 1 ]] && exit 0
if [[ "$FM" == 1 && "$RESUME" == 0 && $(arr PRE) -gt 0 ]]; then
  W="${PRE[0]#weight=}"; [[ -f "$W" ]] || { echo "pretrained checkpoint missing: $W  (see docs/REPRODUCING.md, step 3)"; exit 1; }
fi
H5="$(grep -m1 -E '^H5_PATH\s*=' "$CFG" | sed -E 's/^H5_PATH\s*=\s*"([^"]+)".*/\1/' || true)"
if [[ -n "$H5" && ! -f "$H5" ]]; then echo "dataset file not found: $H5  (see data/DOWNLOAD.md and data/README.md)"; exit 1; fi
command -v python >/dev/null || { echo "no 'python' on PATH: run ./setup_venv.sh and source activate.sh"; exit 1; }
mkdir -p "$OUT/logs"
export CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${CMD[@]}" 2>&1 | tee "$OUT/logs/train_$(date +%Y%m%d_%H%M%S).log"

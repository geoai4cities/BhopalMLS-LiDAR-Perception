#!/bin/bash

################################################################################
# SemanticKITTI Data Structure Preparation Script
# Combines downloaded official KITTI data into the structure required by Pointcept
#
# Source data locations:
#   - data_odometry_velodyne/dataset/sequences/XX/velodyne/  → LiDAR .bin files
#   - data_odometry_color/dataset/sequences/XX/image_2/      → Left color images
#   - data_odometry_color/dataset/sequences/XX/image_3/      → Right color images
#   - data_odometry_labels/dataset/sequences/XX/labels/      → Semantic labels
#   - data_odometry_labels/dataset/sequences/XX/poses.txt    → Vehicle poses
#   - data_odometry_calib/dataset/sequences/XX/calib.txt     → Calibration matrices
#
# Required structure for SemanticKITTIImagePointDataset (9-channel):
#   data_root/dataset/sequences/
#   ├── 00/
#   │   ├── velodyne/          # LiDAR scans (.bin)
#   │   ├── labels/            # Semantic labels (.label)
#   │   ├── image_2/           # Left color camera (.png)
#   │   ├── image_3/           # Right color camera (.png)
#   │   ├── calib.txt          # Calibration matrices
#   │   └── poses.txt          # Vehicle poses
#   ├── 01/
#   └── ...
#
# Usage:
#   ./prepare_semantickitti_9ch.sh
################################################################################

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_info() { echo -e "${BLUE}ℹ${NC} $1"; }

# Directory configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFICIAL_DATA="$SCRIPT_DIR/official"
TARGET_DIR="$SCRIPT_DIR/dataset/sequences"

# Source directories
VELODYNE_SRC="$OFFICIAL_DATA/data_odometry_velodyne/dataset/sequences"
COLOR_SRC="$OFFICIAL_DATA/data_odometry_color/dataset/sequences"
LABELS_SRC="$OFFICIAL_DATA/data_odometry_labels/dataset/sequences"
CALIB_SRC="$OFFICIAL_DATA/data_odometry_calib/dataset/sequences"

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}     SemanticKITTI 9-Channel Data Structure Preparation          ${BLUE}║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Verify source data exists
print_info "Verifying source data..."

if [ ! -d "$VELODYNE_SRC" ]; then
    print_error "Velodyne data not found at: $VELODYNE_SRC"
    exit 1
fi
print_success "Velodyne data found"

if [ ! -d "$COLOR_SRC" ]; then
    print_error "Color data not found at: $COLOR_SRC"
    exit 1
fi
print_success "Color data found"

if [ ! -d "$LABELS_SRC" ]; then
    print_error "Labels data not found at: $LABELS_SRC"
    exit 1
fi
print_success "Labels data found"

if [ ! -d "$CALIB_SRC" ]; then
    print_error "Calibration data not found at: $CALIB_SRC"
    exit 1
fi
print_success "Calibration data found"

echo ""
print_info "Creating target directory structure..."
mkdir -p "$TARGET_DIR"

# Sequences to process (00-21)
SEQUENCES=(00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21)

echo ""
print_info "Processing ${#SEQUENCES[@]} sequences..."
echo ""

for SEQ in "${SEQUENCES[@]}"; do
    echo -e "${BLUE}Processing sequence $SEQ...${NC}"

    SEQ_DIR="$TARGET_DIR/$SEQ"
    mkdir -p "$SEQ_DIR"

    # Link velodyne data
    if [ -d "$VELODYNE_SRC/$SEQ/velodyne" ]; then
        if [ ! -e "$SEQ_DIR/velodyne" ]; then
            ln -sf "$VELODYNE_SRC/$SEQ/velodyne" "$SEQ_DIR/velodyne"
            print_success "  Linked velodyne/"
        else
            print_warning "  velodyne/ already exists"
        fi
    else
        print_error "  velodyne/ not found in source"
    fi

    # Link labels data
    if [ -d "$LABELS_SRC/$SEQ/labels" ]; then
        if [ ! -e "$SEQ_DIR/labels" ]; then
            ln -sf "$LABELS_SRC/$SEQ/labels" "$SEQ_DIR/labels"
            print_success "  Linked labels/"
        else
            print_warning "  labels/ already exists"
        fi
    else
        # Test sequences (11-21) don't have labels
        if [ "$SEQ" -lt "11" ]; then
            print_warning "  labels/ not found (train/val sequence)"
        else
            print_info "  labels/ not found (test sequence - expected)"
        fi
    fi

    # Link image_2 (left color camera)
    if [ -d "$COLOR_SRC/$SEQ/image_2" ]; then
        if [ ! -e "$SEQ_DIR/image_2" ]; then
            ln -sf "$COLOR_SRC/$SEQ/image_2" "$SEQ_DIR/image_2"
            print_success "  Linked image_2/"
        else
            print_warning "  image_2/ already exists"
        fi
    else
        print_error "  image_2/ not found in source"
    fi

    # Link image_3 (right color camera)
    if [ -d "$COLOR_SRC/$SEQ/image_3" ]; then
        if [ ! -e "$SEQ_DIR/image_3" ]; then
            ln -sf "$COLOR_SRC/$SEQ/image_3" "$SEQ_DIR/image_3"
            print_success "  Linked image_3/"
        else
            print_warning "  image_3/ already exists"
        fi
    else
        print_error "  image_3/ not found in source"
    fi

    # Link/copy calib.txt (prefer from color data as it has P2, P3 matrices)
    if [ -f "$COLOR_SRC/$SEQ/calib.txt" ]; then
        if [ ! -e "$SEQ_DIR/calib.txt" ]; then
            ln -sf "$COLOR_SRC/$SEQ/calib.txt" "$SEQ_DIR/calib.txt"
            print_success "  Linked calib.txt (from color data)"
        else
            print_warning "  calib.txt already exists"
        fi
    elif [ -f "$CALIB_SRC/$SEQ/calib.txt" ]; then
        if [ ! -e "$SEQ_DIR/calib.txt" ]; then
            ln -sf "$CALIB_SRC/$SEQ/calib.txt" "$SEQ_DIR/calib.txt"
            print_success "  Linked calib.txt (from calib data)"
        else
            print_warning "  calib.txt already exists"
        fi
    else
        print_error "  calib.txt not found in source"
    fi

    # Link poses.txt
    if [ -f "$LABELS_SRC/$SEQ/poses.txt" ]; then
        if [ ! -e "$SEQ_DIR/poses.txt" ]; then
            ln -sf "$LABELS_SRC/$SEQ/poses.txt" "$SEQ_DIR/poses.txt"
            print_success "  Linked poses.txt"
        else
            print_warning "  poses.txt already exists"
        fi
    else
        # Test sequences (11-21) don't have poses
        if [ "$SEQ" -lt "11" ]; then
            print_warning "  poses.txt not found (train/val sequence)"
        else
            print_info "  poses.txt not found (test sequence - expected)"
        fi
    fi

    echo ""
done

# Verification
echo ""
print_info "Verification Summary:"
echo ""

echo "Structure created at: $TARGET_DIR"
echo ""

# Count complete sequences for training/validation (00-10)
COMPLETE_TRAIN_VAL=0
for SEQ in 00 01 02 03 04 05 06 07 08 09 10; do
    SEQ_DIR="$TARGET_DIR/$SEQ"
    if [ -e "$SEQ_DIR/velodyne" ] && [ -e "$SEQ_DIR/labels" ] && \
       [ -e "$SEQ_DIR/image_2" ] && [ -e "$SEQ_DIR/image_3" ] && \
       [ -e "$SEQ_DIR/calib.txt" ]; then
        COMPLETE_TRAIN_VAL=$((COMPLETE_TRAIN_VAL + 1))
    fi
done

echo "Train/Val sequences ready (00-10): $COMPLETE_TRAIN_VAL / 11"

# Check validation sequence specifically
SEQ_DIR="$TARGET_DIR/08"
if [ -e "$SEQ_DIR/velodyne" ] && [ -e "$SEQ_DIR/labels" ] && \
   [ -e "$SEQ_DIR/image_2" ] && [ -e "$SEQ_DIR/image_3" ] && \
   [ -e "$SEQ_DIR/calib.txt" ]; then
    print_success "Validation sequence 08: READY"
else
    print_error "Validation sequence 08: INCOMPLETE"
fi

# Sample file counts
echo ""
print_info "Sample file counts (sequence 00):"
SEQ00_DIR="$TARGET_DIR/00"
if [ -d "$SEQ00_DIR/velodyne" ]; then
    VELO_COUNT=$(ls "$SEQ00_DIR/velodyne/"*.bin 2>/dev/null | wc -l)
    echo "  velodyne: $VELO_COUNT files"
fi
if [ -d "$SEQ00_DIR/labels" ]; then
    LABEL_COUNT=$(ls "$SEQ00_DIR/labels/"*.label 2>/dev/null | wc -l)
    echo "  labels:   $LABEL_COUNT files"
fi
if [ -d "$SEQ00_DIR/image_2" ]; then
    IMG2_COUNT=$(ls "$SEQ00_DIR/image_2/"*.png 2>/dev/null | wc -l)
    echo "  image_2:  $IMG2_COUNT files"
fi
if [ -d "$SEQ00_DIR/image_3" ]; then
    IMG3_COUNT=$(ls "$SEQ00_DIR/image_3/"*.png 2>/dev/null | wc -l)
    echo "  image_3:  $IMG3_COUNT files"
fi

echo ""
print_info "Data root for training:"
echo "  DATA_ROOT=$SCRIPT_DIR"
echo ""
print_info "Expected dataset path:"
echo "  \$DATA_ROOT/dataset/sequences/XX/..."
echo ""

if [ $COMPLETE_TRAIN_VAL -eq 11 ]; then
    print_success "Data structure preparation COMPLETE!"
    echo ""
    echo "You can now run 9-channel linear probing:"
    echo "  cd Concerto/SemanticKITTI"
    echo "  ./scripts/train_semantickitti_40gb.sh lp 9ch"
else
    print_warning "Some sequences may be incomplete. Check the output above."
fi

echo ""

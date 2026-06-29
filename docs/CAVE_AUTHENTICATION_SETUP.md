# CAVE Authentication Setup for Real Data Pipeline

## Overview

The Neuronauts pipeline can fetch real MICrONS connectome data from CAVE (Connectome Annotation Versioning Engine). While the public `minnie65_public` datastack should technically not require authentication, the current `caveclient` library version requires authentication setup for all requests.

## Current Status

✅ **Synthetic data pipeline**: Working without CAVE access
⚠️ **Real CAVE data**: Requires authentication token setup

## Solution: Setting Up CAVE Authentication

### Step 1: Create CAVE Authentication Token

Follow these steps to get a CAVE API token:

1. Visit the CAVE instance for minnie65_public: https://global.daf-apis.com/
2. Look for authentication/API token generation (usually in user settings or API documentation)
3. Generate a token for the `minnie65_public` datastack

**Alternative (if you have research access)**: Contact the MICrONS consortium for authentication credentials

### Step 2: Configure caveclient

Create a credentials configuration file:

```bash
mkdir -p ~/.caveclient
cat > ~/.caveclient/secrets.json << 'EOF'
{
    "global.daf-apis.com": {
        "minnie65_public": "YOUR_TOKEN_HERE"
    }
}
EOF
chmod 600 ~/.caveclient/secrets.json
```

### Step 3: Test Authentication

```bash
python3 << 'PYEOF'
from caveclient import CAVEclient

try:
    client = CAVEclient("minnie65_public", server_address="https://global.daf-apis.com")
    print(f"✓ Authentication successful!")
    print(f"Datastack: {client.datastack}")
    print(f"Version: {client.version}")
except Exception as e:
    print(f"✗ Authentication failed: {e}")
PYEOF
```

## Fetching Real CAVE Data

Once authenticated, fetch real MICrONS boxes:

### Option 1: Synapse-Seeded Strategy (Recommended)

Fastest method - queries CAVE for real synapse positions and builds boxes around them:

```bash
python scripts/fetch_cave_boxes.py \
  --cache-dir data/boxes_cave \
  --n-boxes 80 \
  --box-side-um 30.0 \
  --no-em \
  --min-positive-pairs 5
```

**Pros**:
- No token required (for public datastack)
- Fast (~1-2 min per box)
- Guarantees non-empty boxes with real synapses

**Cons**:
- Requires CAVE API authentication

### Option 2: CLI Alternative

```bash
python scripts/train.py build-dataset \
  --cache-dir data/boxes_cave \
  --n-boxes 80 \
  --strategy synapse-seeded \
  --no-em \
  --min-positive-pairs 5 \
  --cave-version 1412
```

### Option 3: With EM Volume (Full Data)

If you need electron microscopy volumes for GAT/agent training:

```bash
python scripts/fetch_cave_boxes.py \
  --cache-dir data/boxes_cave_full \
  --n-boxes 30 \
  --box-side-um 30.0 \
  --with-em \
  --min-positive-pairs 5
```

**Warning**: This downloads ~5 GB per box. Recommended for GPU cluster use only.

## Running the Full Pipeline with Real Data

Once you have boxes cached:

### 1. Train Grammar (Synapse-based merge scoring)

```bash
python scripts/train.py train \
  --cache-dir data/boxes_cave \
  --grammar-output models/grammar_microns.pt \
  --epochs 30 \
  --log-dir run_logs/grammar_microns
```

### 2. Train CellGNN (Topological cell reconstruction)

```bash
python scripts/train.py train-cell-gnn \
  --cache-dir data/boxes_cave \
  --epochs 50 \
  --d-model 64 \
  --n-layers 3 \
  --cell-gnn-output models/cell_gnn_microns.pt \
  --log-dir run_logs/cell_gnn_microns
```

### 3. Evaluate

```bash
python scripts/train.py evaluate \
  --cache-dir data/boxes_cave \
  --grammar-path models/grammar_microns.pt \
  --cell-gnn-path models/cell_gnn_microns.pt \
  --output-metrics results/metrics.json
```

## Debugging

### AuthException on Synapse Query

```
caveclient.base.AuthException: You have not setup a token to access
https://global.daf-apis.com/info/api/v2/datastack/full/minnie65_public
```

**Fix**: Ensure `~/.caveclient/secrets.json` is properly configured

### No Boxes Fetched

If `build-dataset` returns 0 boxes:
1. Check network connectivity
2. Verify CAVE credentials are valid
3. Try with lower `--min-positive-pairs` (default: 5)
4. Check that box center is within Minnie65 bounds

### Slow Fetching

- Use `--no-em` to skip EM volume download (10x faster)
- Increase `--box-side-um` to reduce number of queries
- Run multiple fetch jobs in parallel

## Current Working Setup

For development without authentication, use the included synthetic data:

```bash
# Use synthetic boxes (8 boxes, no CAVE needed)
python scripts/train.py train \
  --cache-dir data/boxes_synthetic \
  --grammar-output models/grammar_synthetic.pt \
  --epochs 10
```

## Resources

- CAVEclient Documentation: https://caveconnectome.github.io/CAVEclient/
- CAVE API: https://global.daf-apis.com/info/
- MICrONS Dataset: https://microns-explorer.org/
- Neuronauts README: See ../README.md for full pipeline documentation

## Support

If you have issues with CAVE authentication:
1. Check caveclient version: `pip show caveclient`
2. Review caveclient auth docs: https://caveconnectome.github.io/CAVEclient/tutorials/authentication/
3. Test direct HTTP access to CAVE API
4. Contact MICrONS consortium for research access

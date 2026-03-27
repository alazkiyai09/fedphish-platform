# FedPhish Architecture

## System Overview

FedPhish is a federated learning system designed for collaborative phishing detection among financial institutions. The architecture prioritizes privacy, security, and robustness while maintaining high detection accuracy.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FedPhish System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   Bank A     │    │   Bank B     │    │   Bank C     │     │
│  │              │    │              │    │              │     │
│  │  ┌────────┐  │    │  ┌────────┐  │    │  ┌────────┐  │     │
│  │  │ Local  │  │    │  │ Local  │  │    │  │ Local  │  │     │
│  │  │ Data   │  │    │  │ Data   │  │    │  │ Data   │  │     │
│  │  └────────┘  │    │  └────────┘  │    │  └────────┘  │     │
│  │       │       │    │       │       │    │       │       │     │
│  │  ┌────────▼───────────────────────▼──────────────▼───┐  │     │
│  │  │              Local Training Loop                  │  │     │
│  │  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  │  │     │
│  │  │  │DistilBERT  │  │   LoRA     │  │  XGBoost   │  │  │     │
│  │  │  │  + LoRA    │  │ Adapters   │  │ Features   │  │  │     │
│  │  │  └────────────┘  └────────────┘  └────────────┘  │  │     │
│  │  └───────────────────────────────────────────────────┘  │     │
│  │       │                                                  │     │
│  │  ┌────▼─────────────────────────────────────────────┐   │     │
│  │  │           Privacy Module                          │   │     │
│  │  │  ┌─────────┐  ┌─────────┐  ┌─────────────────┐   │   │     │
│  │  │  │   DP    │→ │    HE   │→ │     ZK Proof    │   │   │     │
│  │  │  │ (Local) │  │ (TenSEAL)│  │   Generation    │   │   │     │
│  │  │  └─────────┘  └─────────┘  └─────────────────┘   │   │     │
│  │  └───────────────────────────────────────────────────┘   │     │
│  └──────┬───────────────────────────────────────────────────┘     │
│         │ Encrypted Update + ZK Proof                            │
│         └─────────────────────────────────────────────┐         │
│                                                           │         │
│  ┌────────────────────────────────────────────────────────▼──┐  │
│  │                    Aggregation Server                       │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │  │
│  │  │  ZK Verify   │→ │  Byzantine   │→ │   HT2ML      │     │  │
│  │  │   Module     │  │   Defense    │  │  Aggregator  │     │  │
│  │  │              │  │              │  │              │     │  │
│  │  │ -Proof Check │  │ -FoolsGold  │  │ -HE (Linear) │     │  │
│  │  │ -Bounds      │  │ -Krum       │  │ -TEE (NonLin)│     │  │
│  │  │ -Validity    │  │ -Trimmed    │  │ -Hybrid      │     │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘     │  │
│  │         │                                            │      │  │
│  │  ┌──────▼───────────────────────────────────────────┐  │   │  │
│  │  │          Reputation System                         │  │   │  │
│  │  │  - Track bank reliability                         │  │   │  │
│  │  │  - Adjust aggregation weights                      │  │   │  │
│  │  │  - Historical contribution scoring                 │  │   │  │
│  │  └────────────────────────────────────────────────────┘  │   │  │
│  │         │                                                  │   │  │
│  │  ┌──────▼───────────────────────────────────────────┐    │   │  │
│  │  │           Global Model Update                     │    │   │  │
│  │  │  - Aggregated parameters                          │    │   │  │
│  │  │  - Distributed to all banks                       │    │   │  │
│  │  └────────────────────────────────────────────────────┘    │   │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Client Module

#### Model Wrapper (`client/model.py`)
- **FedPhishModel**: Wrapper around DistilBERT with LoRA
- Implements Flower framework interface
- Parameter serialization/deserialization
- Model update methods

#### Privacy Engine (`client/privacy.py`)
- **ClientPrivacyEngine**: Applies privacy before transmission
- Local DP with gradient clipping
- HE encryption (CKKS via TenSEAL)
- Configurable privacy levels (1-3)

#### ZK Prover (`client/prover.py`)
- **GradientBoundsProver**: Generates ZK proofs
- Proves gradient norms within bounds
- Batch proof generation
- Local verification

#### Trainer (`client/trainer.py`)
- **FedPhishClient**: Main Flower client
- Local training loop
- Privacy-aware update generation
- Evaluation hook

### 2. Server Module

#### Aggregator (`server/aggregator.py`)
- **HT2MLAggregator**: Hybrid HE+TEE aggregation
- Linear operations via HE
- Non-linear operations via TEE
- Multiple aggregation strategies

#### Verifier (`server/verifier.py`)
- **ProofVerifier**: Verifies ZK proofs
- Batch verification
- Proof validation
- Invalid update filtering

#### Defense (`server/defense.py`)
- **ByzantineDetector**: Detects malicious clients
- Multiple defense strategies:
  - FoolsGold (similarity-based)
  - Krum (distance-based)
  - Trimmed Mean
  - Combined (multi-strategy)
- Adaptive threat response

#### Reputation (`server/reputation.py`)
- **BankReputation**: Tracks bank reliability
- Factors: proof validity, similarity, contribution
- EMA-based updates
- Weight computation for aggregation

### 3. Detection Module

#### Feature Extraction (`detection/features.py`)
- **URLFeatureExtractor**: Lexical + host-based features
- **EmailFeatureExtractor**: Header + body features
- **TFIDFVectorizer**: Text vectorization
- **FeatureFusion**: Combines multiple feature types

#### Transformer (`detection/transformer.py`)
- **DistilBERTPhishing**: Pre-trained DistilBERT
- **LoRAAdapter**: Efficient fine-tuning
- **FeatureExtractor**: Embedding extraction
- **AttentionVisualization**: Attention weight visualization

#### Ensemble (`detection/ensemble.py`)
- **HybridEnsemble**: DistilBERT + XGBoost
- **WeightedCombiner**: Learns optimal combination
- Trainable combination weights

#### Explainer (`detection/explainer.py`)
- **SHAPExplainer**: SHAP-based explanations
- **AttentionVisualizer**: Attention heatmap
- **ReportGenerator**: Automated reports

### 4. Privacy Module

#### Differential Privacy (`privacy/dp.py`)
- **GradientClipper**: L2 norm clipping
- **DifferentialPrivacy**: Gaussian mechanism
- **PrivacyAccountant**: RDP accounting
- **AdaptivePrivacy**: Adaptive noise

#### Homomorphic Encryption (`privacy/he.py`)
- **TenSEALContext**: CKKS encryption context
- **EncryptedGradient**: Encrypted gradient wrapper
- **SecureAggregator**: HE-based aggregation

#### Trusted Execution Environment (`privacy/tee.py`)
- **GramineTEE**: TEE simulation
- **TrustedAggregator**: Non-linear ops in TEE
- **Attestation**: Remote attestation

#### HT2ML (`privacy/ht2ml.py`)
- **HT2MLAggregator**: Hybrid framework
- **PrivacyLevel**: 3-level privacy
- **HT2MLClient**: Client-side operations

### 5. Security Module

#### Zero-Knowledge Proofs (`security/zkp.py`)
- **GradientBoundsProof**: ZK proof for bounds
- **ZKProver**: Proof generation
- **ZKVerifier**: Proof verification
- **ProofSystem**: Complete system

#### Attacks (`security/attacks.py`)
- **SignFlipAttack**: Negates gradients
- **GaussianNoiseAttack**: Adds noise
- **BackdoorAttack**: Embeds backdoor
- **LabelFlippingAttack**: Data poisoning
- **AGRAttackResistant**: AGR-specific attacks

#### Defenses (`security/defenses.py`)
- **KrumDefense**: Distance-based defense
- **FoolsGoldDefense**: Similarity-based defense
- **TrimmedMeanDefense**: Removes extremes
- **NormClippingDefense**: Clips gradients
- **CombinedDefense**: Multi-strategy

### 6. Utils Module

#### Data (`utils/data.py`)
- **BankDataPartitioner**: Splits data across banks
- **PhishingDataLoader**: Data loading
- **BankSimulation**: Multi-bank simulation

#### Metrics (`utils/metrics.py`)
- **FederatedMetrics**: FL-specific metrics
- **PrivacyMetrics**: Privacy tracking
- **SecurityMetrics**: Security metrics
- **DetectionMetrics**: Classification metrics

#### Visualization (`utils/visualization.py`)
- **TrainingVisualizer**: Training curves
- **PrivacyVisualizer**: Privacy plots
- **SecurityVisualizer**: Security plots
- **BankVisualizer**: Bank-specific plots

## Data Flow

### Training Flow

1. **Initialization**
   - Server sends global model parameters
   - Each bank loads local data
   - Privacy engine initialized

2. **Local Training (per bank)**
   ```
   Local Data
       ↓
   Train DistilBERT+LoRA
       ↓
   Compute Gradients
       ↓
   [Privacy Pipeline]
       ↓
   Clip Gradients → Add DP Noise → Encrypt (HE) → Generate ZK Proof
       ↓
   Send Update + Proof
   ```

3. **Server Aggregation**
   ```
   Receive Updates + Proofs
       ↓
   [Verification Pipeline]
       ↓
   Verify ZK Proofs → Detect Malicious → Compute Reputation Weights
       ↓
   [Aggregation Pipeline]
       ↓
   Decrypt (HE) → Apply Weights → HT2ML Aggregate
       ↓
   Update Global Model
       ↓
   Broadcast to Banks
   ```

### Inference Flow

```
Input Text (URL/Email)
    ↓
[Feature Extraction]
    ↓
Engineered Features ← → DistilBERT Embeddings
    ↓                      ↓
    └──────→ Hybrid Ensemble ←─────┘
                ↓
           Prediction
                ↓
        [Optional: Explanation]
                ↓
        SHAP Values + Attention
```

## Security Architecture

### Threat Model

**Assumed Threats:**
- Curious server (wants to see raw updates)
- Malicious clients (send corrupted updates)
- External attackers (try to infer training data)

**Defenses:**
1. **Privacy**
   - Local DP prevents gradient inversion
   - HE prevents server from seeing updates
   - TEE protects non-linear operations

2. **Robustness**
   - ZK proofs verify gradient bounds
   - Byzantine defenses filter malicious updates
   - Reputation system downweights unreliable clients

3. **Verifiability**
   - All updates have ZK proofs
   - Attestation for TEE operations
   - Audit logs for all aggregations

## Performance Considerations

### Communication Overhead

| Component | Overhead | Mitigation |
|-----------|----------|------------|
| HE (Level 2) | +50% | Batch encryption |
| TEE (Level 3) | +10% | Async attestation |
| ZK Proofs | +5% | Batch verification |

### Computation Overhead

| Operation | Time (ms) | Optimization |
|-----------|-----------|--------------|
| Local DP | ~10 | Vectorized ops |
| HE Encrypt | ~100 | GPU acceleration |
| HE Decrypt | ~50 | Batch operations |
| ZK Proof Gen | ~200 | Parallelization |
| ZK Verify | ~50 | Batch verify |

### Scalability

- **Banks**: Tested up to 50 banks
- **Data per bank**: 1K - 100K samples
- **Rounds**: Typical 20-50 rounds
- **Convergence**: 10-20 rounds for good accuracy

## Configuration

All components are configured via YAML files:

```yaml
# configs/base.yaml
experiment:
  num_banks: 5
  num_rounds: 20
  privacy_level: 3

privacy:
  epsilon: 1.0
  delta: 1e-5

security:
  enable_zk_proofs: true
  defense_strategy: "foolsgold"
```

See `configs/` directory for all configuration options.

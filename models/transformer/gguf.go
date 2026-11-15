package transformer

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/humanize"
	"github.com/gomlx/go-huggingface/models/gguf"
	"github.com/gomlx/go-huggingface/tokenizers"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// IsGGUF reports whether the model is backed by a GGUF weights file.
func (m *Model) IsGGUF() bool {
	return m.isGGUF
}

// GGUFFile returns the selected .gguf filename within m.Repo (or "" if not a GGUF model).
func (m *Model) GGUFFile() string {
	return m.ggufFile
}

// GGUFFiles returns all .gguf filenames discovered in m.Repo.
func (m *Model) GGUFFiles() []string {
	return m.ggufFiles
}

// WithGGUFFile selects a specific .gguf file in m.Repo and enables GGUF loading mode.
func (m *Model) WithGGUFFile(filename string) *Model {
	if m.ggufFile != filename {
		m.parsedGGUF = nil
		m.totalParameters = nil
		m.totalBytes = nil
	}
	m.ggufFile = filename
	m.isGGUF = filename != ""
	return m
}

// WithApproximateGelu configures whether GELU activations use the fast tanh approximation
// (true, matching GoMLX's default and llama.cpp's ggml_gelu) or the exact erf formulation
// (false, matching PyTorch/ONNX nn.GELU(approximate="none")).
func (m *Model) WithApproximateGelu(approx bool) *Model {
	m.approximateGelu = &approx
	return m
}

// detectRepoFormats inspects repository filenames for .gguf and .safetensors weights.
func (m *Model) detectRepoFormats() (hasSafeTensors bool) {
	for filename, err := range m.Repo.IterFileNames() {
		if err != nil {
			break
		}
		lower := strings.ToLower(filename)
		switch {
		case strings.HasSuffix(lower, ".gguf"):
			m.ggufFiles = append(m.ggufFiles, filename)
		case strings.HasSuffix(lower, ".safetensors") || strings.HasSuffix(lower, ".safetensors.index.json"):
			hasSafeTensors = true
		}
	}
	return hasSafeTensors
}

// ensureGGUFHeader downloads (if not already cached) and parses the selected GGUF file header,
// populating missing model configuration, causal-mask, and pooling metadata.
func (m *Model) ensureGGUFHeader() error {
	if m.parsedGGUF != nil {
		return nil
	}
	if m.ggufFile == "" {
		if len(m.ggufFiles) == 0 {
			m.detectRepoFormats()
		}
		m.ggufFile = gguf.SelectGGUFFile(m.ggufFiles)
	}
	if m.ggufFile == "" {
		return errors.Errorf("transformer: no .gguf file found in repository %q", m.Repo.ID)
	}

	paths, err := m.Repo.DownloadFiles(m.ggufFile)
	if err != nil {
		return errors.Wrapf(err, "transformer: download GGUF file %q", m.ggufFile)
	}
	f, err := gguf.Open(paths[0])
	if err != nil {
		return errors.Wrapf(err, "transformer: parse GGUF file %q", m.ggufFile)
	}
	m.parsedGGUF = f
	m.applyGGUFMetadata(f)
	return nil
}

// applyGGUFMetadata populates missing fields of m.Config, m.PoolingConfig, m.Modules, and
// m.useCausalMask from parsed GGUF header metadata.
func (m *Model) applyGGUFMetadata(f *gguf.File) {
	if f == nil {
		return
	}
	arch := strings.ToLower(strings.TrimSpace(f.Architecture()))
	if m.Config.ModelType == "" && arch != "" {
		m.Config.ModelType = arch
	}
	if m.Config.Extra == nil {
		m.Config.Extra = make(map[string]any)
	}

	getInt := func(suffix string) int {
		if arch == "" {
			return 0
		}
		if kv, ok := f.GetKeyValue(arch + "." + suffix); ok {
			return int(kv.Int64())
		}
		return 0
	}
	getFloat := func(suffix string) float64 {
		if arch == "" {
			return 0
		}
		if kv, ok := f.GetKeyValue(arch + "." + suffix); ok {
			return kv.Float64()
		}
		return 0
	}

	if m.Config.HiddenSize == 0 {
		m.Config.HiddenSize = getInt("embedding_length")
	}
	if m.Config.NumHiddenLayers == 0 {
		m.Config.NumHiddenLayers = getInt("block_count")
	}
	if m.Config.IntermediateSize == 0 {
		m.Config.IntermediateSize = getInt("feed_forward_length")
	}
	if m.Config.NumAttentionHeads == 0 {
		m.Config.NumAttentionHeads = getInt("attention.head_count")
	}
	if m.Config.NumKeyValueHeads == 0 {
		m.Config.NumKeyValueHeads = getInt("attention.head_count_kv")
	}
	if m.Config.MaxPositionEmbeddings == 0 {
		m.Config.MaxPositionEmbeddings = getInt("context_length")
	}
	if m.Config.LayerNormEps == 0 {
		m.Config.LayerNormEps = getFloat("attention.layer_norm_epsilon")
	}
	if m.Config.RMSNormEps == 0 {
		m.Config.RMSNormEps = getFloat("attention.layer_norm_rms_epsilon")
	}
	if m.Config.RoPETheta == 0 {
		m.Config.RoPETheta = getFloat("rope.freq_base")
	}
	if m.Config.VocabSize == 0 {
		if kv, ok := f.GetKeyValue(gguf.KeyTokenizerTokens); ok {
			m.Config.VocabSize = len(kv.Strings())
		} else if ti, ok := f.GetTensorInfo("token_embd.weight"); ok && len(ti.Shape) == 2 {
			m.Config.VocabSize = int(ti.Shape[1])
		}
	}
	if _, hasTypeVocab := m.Config.Extra["type_vocab_size"]; !hasTypeVocab {
		if kv, ok := f.GetKeyValue("tokenizer.ggml.token_type_count"); ok && kv.Int64() > 0 {
			m.Config.Extra["type_vocab_size"] = float64(kv.Int64())
		} else if ti, ok := f.GetTensorInfo("token_types.weight"); ok && len(ti.Shape) == 2 {
			m.Config.Extra["type_vocab_size"] = float64(ti.Shape[1])
		}
	}
	if m.Config.HiddenAct == "" && m.Config.HiddenActivation == "" {
		switch arch {
		case "bert", "nomic-bert", "jina-bert-v2":
			m.Config.HiddenAct = "gelu"
		case "gemma", "gemma2", "gemma3":
			m.Config.HiddenActivation = "gelu_pytorch_tanh"
		default:
			m.Config.HiddenAct = "silu"
		}
	}

	// Pooling type from GGUF (llama_pooling_type: 0=NONE, 1=MEAN, 2=CLS, 3=LAST).
	if m.PoolingConfig == nil && arch != "" {
		if kv, ok := f.GetKeyValue(arch + ".pooling_type"); ok {
			switch kv.Int64() {
			case 1:
				m.PoolingConfig = &PoolingConfig{PoolingModeMeanTokens: true}
			case 2:
				m.PoolingConfig = &PoolingConfig{PoolingModeClsToken: true}
			case 3:
				m.PoolingConfig = &PoolingConfig{PoolingModeLastToken: true}
			}
			if m.PoolingConfig != nil && len(m.Modules) == 0 {
				// Embedding GGUFs (such as sentence-transformers exports) omit modules.json;
				// synthesize the standard Transformer -> Pooling -> L2 Normalize pipeline.
				m.Modules = []ModuleConfig{
					{Idx: 0, Name: "0", Type: "sentence_transformers.models.Transformer"},
					{Idx: 1, Name: "1", Path: "1_Pooling", Type: "sentence_transformers.models.Pooling"},
					{Idx: 2, Name: "2", Path: "2_Normalize", Type: "sentence_transformers.models.Normalize"},
				}
			}
		}
	}

	// Explicit "<arch>.attention.causal" overrides heuristic detection when present.
	if arch != "" {
		if kv, ok := f.GetKeyValue(arch + ".attention.causal"); ok {
			if causal, isBool := kv.Raw().(bool); isBool {
				m.useCausalMask = causal
				return
			}
		}
	}
	m.useCausalMask = m.detectCausalMask()
}

// loadGGUFTokenizer builds a tokenizer from the GGUF file's embedded "tokenizer.ggml.*" metadata.
func (m *Model) loadGGUFTokenizer() (tokenizers.Tokenizer, error) {
	if err := m.ensureGGUFHeader(); err != nil {
		return nil, err
	}
	cfg, _ := tokenizers.GetConfig(m.Repo)
	return m.parsedGGUF.LoadTokenizer(cfg)
}

// loadStoreGGUF loads GGUF tensors into store, keeping supported 2D GGML quantized weights in
// their native compressed Uint8 block layout with model.Quantization metadata.
func ggufFloatDType(t gguf.TensorType) (dtypes.DType, bool) {
	switch t {
	case gguf.TensorTypeF32:
		return dtypes.Float32, true
	case gguf.TensorTypeF16:
		return dtypes.Float16, true
	case gguf.TensorTypeBF16:
		return dtypes.BFloat16, true
	default:
		return dtypes.InvalidDType, false
	}
}

// ggufDTypes infers the primary model computation dtype and normalization parameter dtype
// from the GGUF tensor headers.
//
// If ForceFloat32() is enabled or the GGUF contains any quantized tensors, both dtypes
// default to Float32 (keeping quantized weights via QuantGGML and promoting small unquantized
// F16/BF16 tensors such as position embeddings to Float32). For unquantized half-precision
// GGUFs, the dominant 2D weight dtype becomes modelDType while normDType tracks the
// normalization weights (often Float32 in llama.cpp F16/BF16 exports).
func (m *Model) ggufDTypes() (modelDType, normDType dtypes.DType) {
	if m.forceFloat32 {
		return dtypes.Float32, dtypes.Float32
	}
	if err := m.ensureGGUFHeader(); err != nil || m.parsedGGUF == nil {
		return dtypes.Float32, dtypes.Float32
	}

	isBert := m.isBertModel()
	for _, info := range m.parsedGGUF.TensorInfos {
		if _, quantOK := info.Type.ToGGMLQuantType(); quantOK {
			return dtypes.Float32, dtypes.Float32
		}
	}

	matrixElems := make(map[dtypes.DType]uint64)
	otherElems := make(map[dtypes.DType]uint64)
	var seenNormF32, seenNormF16, seenNormBF16 bool

	for _, info := range m.parsedGGUF.TensorInfos {
		dt, ok := ggufFloatDType(info.Type)
		if !ok {
			continue
		}

		isNorm := false
		if _, isWeight, isQKV := parseGGUFFusedQKV(info.Name); isQKV {
			_ = isWeight
		} else {
			_, varName, mapped := mapGGUFTensorName(info.Name, isBert)
			if !mapped {
				continue
			}
			isNorm = varName == "gain" || varName == "offset" || varName == "scale"
		}

		if isNorm {
			switch dt {
			case dtypes.Float32:
				seenNormF32 = true
			case dtypes.Float16:
				seenNormF16 = true
			case dtypes.BFloat16:
				seenNormBF16 = true
			}
			continue
		}

		elems := info.NumElements()
		if len(info.Shape) == 2 {
			matrixElems[dt] += elems
		} else {
			otherElems[dt] += elems
		}
	}

	pickDominant := func(counts map[dtypes.DType]uint64) dtypes.DType {
		bestDT := dtypes.InvalidDType
		var bestCount uint64
		for _, candidate := range []dtypes.DType{dtypes.Float32, dtypes.Float16, dtypes.BFloat16} {
			if c := counts[candidate]; c > bestCount {
				bestCount = c
				bestDT = candidate
			}
		}
		return bestDT
	}

	modelDType = pickDominant(matrixElems)
	if modelDType == dtypes.InvalidDType {
		modelDType = pickDominant(otherElems)
	}
	if modelDType == dtypes.InvalidDType {
		modelDType = dtypes.Float32
	}

	normCount := 0
	if seenNormF32 {
		normCount++
		normDType = dtypes.Float32
	}
	if seenNormF16 {
		normCount++
		normDType = dtypes.Float16
	}
	if seenNormBF16 {
		normCount++
		normDType = dtypes.BFloat16
	}
	if normCount == 0 {
		normDType = modelDType
	} else if normCount > 1 {
		normDType = dtypes.Float32
	}

	return modelDType, normDType
}

func (m *Model) loadStoreGGUF(backend compute.Backend, store *model.Store, filterFn func(path string) bool) error {
	if err := m.ensureGGUFHeader(); err != nil {
		return err
	}
	reader, err := gguf.NewReader(m.parsedGGUF)
	if err != nil {
		return err
	}
	defer reader.Close()

	isBert := m.isBertModel()
	targetModelDType, targetNormDType := m.ggufDTypes()
	var totalParams int64
	var totalBytes int64

	storeLoadedVar := func(scopePath []string, varName string, t *tensors.Tensor, q *model.Quantization, logicalElems, byteSize int64) {
		fullPath := "/" + strings.Join(append(append([]string(nil), scopePath...), varName), "/")
		if !filterFn(fullPath) || store.GetVariable(fullPath) != nil {
			t.FinalizeAll()
			return
		}
		totalParams += logicalElems
		totalBytes += byteSize

		subScope := store.RootScope()
		for _, subScopeName := range scopePath {
			subScope = subScope.In("%s", subScopeName)
		}
		v := subScope.VariableWithValue(varName, t)
		if q != nil {
			v.WithQuantization(q)
		}
	}

	for _, info := range m.parsedGGUF.TensorInfos {
		// Handle fused QKV tensors (e.g. blk.{N}.attn_qkv.{weight,bias} in some BERT GGUF exports).
		if layerNum, isWeight, isQKV := parseGGUFFusedQKV(info.Name); isQKV {
			if err := loadSplitGGUFQKV(reader, backend, info, layerNum, isBert, isWeight, targetModelDType == dtypes.Float32, storeLoadedVar); err != nil {
				return err
			}
			continue
		}

		scopePath, varName, ok := mapGGUFTensorName(info.Name, isBert)
		if !ok {
			klog.V(1).Infof("Skipping unmapped GGUF tensor %q in %s", info.Name, filepath.Base(m.ggufFile))
			continue
		}

		fullPath := "/" + strings.Join(append(append([]string(nil), scopePath...), varName), "/")
		if !filterFn(fullPath) || store.GetVariable(fullPath) != nil {
			continue
		}

		if ggmlType, quantOK := info.Type.ToGGMLQuantType(); quantOK && len(info.Shape) == 2 {
			rawTensor, err := reader.ReadTensorRaw(backend, info.Name)
			if err != nil {
				return errors.WithMessagef(err, "failed loading raw GGUF tensor %q from %q", info.Name, m.ggufFile)
			}
			q := &model.Quantization{
				Scheme:   compute.QuantGGML,
				GGMLType: ggmlType,
			}
			storeLoadedVar(scopePath, varName, rawTensor, q, int64(info.NumElements()), info.NumBytes())
			continue
		}

		// Native float/int tensor (or non-2D / fallback quantized tensor): load on host first so
		// Float32 promotion can be applied when target dtype requires Float32 (quantized models,
		// mixed norm dtypes, Float64 tensors, or ForceFloat32).
		hostTensor, err := reader.ReadTensor(nil, info.Name)
		if err != nil {
			return errors.WithMessagef(err, "failed loading GGUF tensor %q from %q", info.Name, m.ggufFile)
		}
		isNorm := varName == "gain" || varName == "offset" || varName == "scale"
		needFloat32 := targetModelDType == dtypes.Float32 ||
			hostTensor.DType() == dtypes.Float64 ||
			(isNorm && targetNormDType == dtypes.Float32)
		tensorToLoad := hostTensor
		if needFloat32 {
			tensorToLoad, err = ensureFloat32HostTensor(hostTensor)
			if err != nil {
				hostTensor.FinalizeAll()
				return errors.WithMessagef(err, "failed converting GGUF tensor %q to Float32", info.Name)
			}
		}
		if backend != nil {
			if err := tensorToLoad.ToDevice(backend, 0); err != nil {
				tensorToLoad.FinalizeAll()
				return errors.WithMessagef(err, "failed moving GGUF tensor %q to backend device #0", info.Name)
			}
		}
		shape := tensorToLoad.Shape()
		storeLoadedVar(scopePath, varName, tensorToLoad, nil, int64(shape.Size()), int64(shape.ByteSize()))
	}

	// Tie decoder output weights to token_embed/embeddings when output.weight is omitted.
	if !isBert && store.GetVariable("/output/dense/weights") == nil && filterFn("/output/dense/weights") {
		if embedVar := store.GetVariable("/token_embed/embeddings"); embedVar != nil {
			embedVal, err := embedVar.Value()
			if err != nil {
				return err
			}
			cloned, err := embedVal.Clone()
			if err != nil {
				return err
			}
			outVar := store.RootScope().In("output").In("dense").VariableWithValue("weights", cloned)
			if embedVar.Quantization() != nil {
				qCopy := *embedVar.Quantization()
				outVar.WithQuantization(&qCopy)
			}
		}
	}

	klog.V(1).Infof("loadStoreGGUF (%s): Loaded %s logical parameters (%s compressed/in-memory bytes)",
		filepath.Base(m.ggufFile), humanize.Count(totalParams), humanize.Bytes(totalBytes))
	m.totalParameters = &totalParams
	m.totalBytes = &totalBytes
	return nil
}

func (m *Model) isBertModel() bool {
	mt := strings.ToLower(strings.TrimSpace(m.Config.ModelType))
	if mt == "bert" || mt == "nomic-bert" || mt == "jina-bert-v2" {
		return true
	}
	return len(m.Config.Architectures) > 0 && m.Config.Architectures[0] == "BertModel"
}

// mapGGUFTensorName maps GGUF canonical tensor names to GoMLX Scope paths and variable names.
func mapGGUFTensorName(ggufName string, isBert bool) (scopePath []string, varName string, ok bool) {
	switch ggufName {
	case "token_embd.weight":
		return []string{"token_embed"}, "embeddings", true
	case "position_embd.weight":
		return []string{"pos_embed"}, "embeddings", true
	case "token_types.weight":
		return []string{"token_type_embed"}, "embeddings", true
	case "token_embd_norm.weight":
		if isBert {
			return []string{"embed_norm", "layer_normalization"}, "gain", true
		}
		return []string{"embed_norm", "rms_norm"}, "scale", true
	case "token_embd_norm.bias":
		return []string{"embed_norm", "layer_normalization"}, "offset", true
	case "output_norm.weight":
		if isBert {
			return []string{"final_norm", "layer_normalization"}, "gain", true
		}
		return []string{"final_norm", "rms_norm"}, "scale", true
	case "output_norm.bias":
		return []string{"final_norm", "layer_normalization"}, "offset", true
	case "output.weight":
		return []string{"output", "dense"}, "weights", true
	case "output.bias":
		return []string{"output", "dense"}, "biases", true
	}

	var layerNum int
	var rest string
	if n, err := fmt.Sscanf(ggufName, "blk.%d.%s", &layerNum, &rest); n == 2 && err == nil {
		layerScope := fmt.Sprintf("layer_%d", layerNum)
		if isBert {
			switch rest {
			case "attn_q.weight":
				return []string{layerScope, "attn", "MultiHeadAttention", "query", "dense"}, "weights", true
			case "attn_q.bias":
				return []string{layerScope, "attn", "MultiHeadAttention", "query", "dense"}, "biases", true
			case "attn_k.weight":
				return []string{layerScope, "attn", "MultiHeadAttention", "key", "dense"}, "weights", true
			case "attn_k.bias":
				return []string{layerScope, "attn", "MultiHeadAttention", "key", "dense"}, "biases", true
			case "attn_v.weight":
				return []string{layerScope, "attn", "MultiHeadAttention", "value", "dense"}, "weights", true
			case "attn_v.bias":
				return []string{layerScope, "attn", "MultiHeadAttention", "value", "dense"}, "biases", true
			case "attn_output.weight":
				return []string{layerScope, "attn", "MultiHeadAttention", "output", "dense"}, "weights", true
			case "attn_output.bias":
				return []string{layerScope, "attn", "MultiHeadAttention", "output", "dense"}, "biases", true
			case "attn_output_norm.weight", "attn_norm.weight":
				return []string{layerScope, "norm1", "layer_normalization"}, "gain", true
			case "attn_output_norm.bias", "attn_norm.bias":
				return []string{layerScope, "norm1", "layer_normalization"}, "offset", true
			case "ffn_up.weight":
				return []string{layerScope, "ff1", "dense"}, "weights", true
			case "ffn_up.bias":
				return []string{layerScope, "ff1", "dense"}, "biases", true
			case "ffn_down.weight":
				return []string{layerScope, "ff2", "dense"}, "weights", true
			case "ffn_down.bias":
				return []string{layerScope, "ff2", "dense"}, "biases", true
			case "layer_output_norm.weight", "ffn_norm.weight":
				return []string{layerScope, "norm2", "layer_normalization"}, "gain", true
			case "layer_output_norm.bias", "ffn_norm.bias":
				return []string{layerScope, "norm2", "layer_normalization"}, "offset", true
			}
			return nil, "", false
		}

		// Decoder / Gemma / LLaMA / Qwen block mappings.
		switch rest {
		case "attn_norm.weight":
			return []string{layerScope, "input_norm", "rms_norm"}, "scale", true
		case "post_attention_norm.weight":
			return []string{layerScope, "post_attention_norm", "rms_norm"}, "scale", true
		case "ffn_norm.weight":
			return []string{layerScope, "pre_feedforward_norm", "rms_norm"}, "scale", true
		case "post_ffw_norm.weight":
			return []string{layerScope, "post_feedforward_norm", "rms_norm"}, "scale", true
		case "attn_q.weight":
			return []string{layerScope, "self_attn", "MultiHeadAttention", "query", "dense"}, "weights", true
		case "attn_q.bias":
			return []string{layerScope, "self_attn", "MultiHeadAttention", "query", "dense"}, "biases", true
		case "attn_k.weight":
			return []string{layerScope, "self_attn", "MultiHeadAttention", "key", "dense"}, "weights", true
		case "attn_k.bias":
			return []string{layerScope, "self_attn", "MultiHeadAttention", "key", "dense"}, "biases", true
		case "attn_v.weight":
			return []string{layerScope, "self_attn", "MultiHeadAttention", "value", "dense"}, "weights", true
		case "attn_v.bias":
			return []string{layerScope, "self_attn", "MultiHeadAttention", "value", "dense"}, "biases", true
		case "attn_output.weight":
			return []string{layerScope, "self_attn", "MultiHeadAttention", "output", "dense"}, "weights", true
		case "attn_output.bias":
			return []string{layerScope, "self_attn", "MultiHeadAttention", "output", "dense"}, "biases", true
		case "attn_q_norm.weight":
			return []string{layerScope, "self_attn", "MultiHeadAttention", "query", "rms_norm"}, "scale", true
		case "attn_k_norm.weight":
			return []string{layerScope, "self_attn", "MultiHeadAttention", "key", "rms_norm"}, "scale", true
		case "ffn_gate.weight":
			return []string{layerScope, "mlp", "gate_proj", "dense"}, "weights", true
		case "ffn_gate.bias":
			return []string{layerScope, "mlp", "gate_proj", "dense"}, "biases", true
		case "ffn_up.weight":
			return []string{layerScope, "mlp", "up_proj", "dense"}, "weights", true
		case "ffn_up.bias":
			return []string{layerScope, "mlp", "up_proj", "dense"}, "biases", true
		case "ffn_down.weight":
			return []string{layerScope, "mlp", "down_proj", "dense"}, "weights", true
		case "ffn_down.bias":
			return []string{layerScope, "mlp", "down_proj", "dense"}, "biases", true
		}
	}
	return nil, "", false
}

func parseGGUFFusedQKV(name string) (layerNum int, isWeight bool, ok bool) {
	var rest string
	if n, err := fmt.Sscanf(name, "blk.%d.%s", &layerNum, &rest); n == 2 && err == nil {
		switch rest {
		case "attn_qkv.weight":
			return layerNum, true, true
		case "attn_qkv.bias":
			return layerNum, false, true
		}
	}
	return 0, false, false
}

func loadSplitGGUFQKV(
	reader *gguf.Reader,
	backend compute.Backend,
	info gguf.TensorInfo,
	layerNum int,
	isBert bool,
	isWeight bool,
	forceFloat32 bool,
	storeFn func(scopePath []string, varName string, t *tensors.Tensor, q *model.Quantization, logicalElems, byteSize int64),
) error {
	layerScope := fmt.Sprintf("layer_%d", layerNum)
	attnGroup := "self_attn"
	if isBert {
		attnGroup = "attn"
	}
	targets := []string{"query", "key", "value"}

	if isWeight {
		if ggmlType, quantOK := info.Type.ToGGMLQuantType(); quantOK && len(info.Shape) == 2 && info.Shape[1]%3 == 0 {
			raw, err := reader.ReadTensorRaw(nil, info.Name)
			if err != nil {
				return err
			}
			defer raw.FinalizeAll()

			totalRows := raw.Shape().Dimensions[0]
			rowsPerPart := totalRows / 3
			bytesPerRow := raw.Shape().Dimensions[1]
			partBytes := rowsPerPart * bytesPerRow
			elemsPerPart := int64(info.NumElements()) / 3

			var rawSlice []byte
			if err := raw.ConstBytes(func(b []byte) {
				rawSlice = append([]byte(nil), b...)
			}); err != nil {
				return err
			}
			for idx, part := range targets {
				partTensor := tensors.FromShape(shapes.Make(dtypes.Uint8, rowsPerPart, bytesPerRow))
				_ = partTensor.MutableBytes(func(dst []byte) {
					copy(dst, rawSlice[idx*partBytes:(idx+1)*partBytes])
				})
				if backend != nil {
					if err := partTensor.ToDevice(backend, 0); err != nil {
						partTensor.FinalizeAll()
						return err
					}
				}
				storeFn(
					[]string{layerScope, attnGroup, "MultiHeadAttention", part, "dense"},
					"weights",
					partTensor,
					&model.Quantization{Scheme: compute.QuantGGML, GGMLType: ggmlType},
					elemsPerPart,
					int64(partBytes),
				)
			}
			return nil
		}
	}

	hostT, err := reader.ReadTensor(nil, info.Name)
	if err != nil {
		return err
	}
	if forceFloat32 || hostT.DType() == dtypes.Float64 {
		converted, err := ensureFloat32HostTensor(hostT)
		if err != nil {
			hostT.FinalizeAll()
			return err
		}
		hostT = converted
	}
	defer hostT.FinalizeAll()

	dims := hostT.Shape().Dimensions
	var varName string
	var partDims []int
	if isWeight {
		if len(dims) != 2 || dims[0]%3 != 0 {
			return errors.Errorf("unexpected shape %s for %s", hostT.Shape(), info.Name)
		}
		varName = "weights"
		partDims = []int{dims[0] / 3, dims[1]}
	} else {
		if len(dims) != 1 || dims[0]%3 != 0 {
			return errors.Errorf("unexpected shape %s for %s", hostT.Shape(), info.Name)
		}
		varName = "biases"
		partDims = []int{dims[0] / 3}
	}

	parts, err := splitTensorFirstAxis3(hostT, partDims...)
	if err != nil {
		return err
	}
	for idx, part := range targets {
		partTensor := parts[idx]
		if backend != nil {
			if err := partTensor.ToDevice(backend, 0); err != nil {
				for _, rem := range parts[idx:] {
					rem.FinalizeAll()
				}
				return err
			}
		}
		partShape := partTensor.Shape()
		storeFn(
			[]string{layerScope, attnGroup, "MultiHeadAttention", part, "dense"},
			varName,
			partTensor,
			nil,
			int64(partShape.Size()),
			int64(partShape.ByteSize()),
		)
	}
	return nil
}

func splitTensorFirstAxis3(t *tensors.Tensor, partDims ...int) ([]*tensors.Tensor, error) {
	switch t.DType() {
	case dtypes.Float32:
		return splitTypedFirstAxis3[float32](t, partDims...)
	case dtypes.Float16:
		return splitTypedFirstAxis3[float16.Float16](t, partDims...)
	case dtypes.BFloat16:
		return splitTypedFirstAxis3[bfloat16.BFloat16](t, partDims...)
	default:
		return nil, errors.Errorf("unsupported dtype %s for fused QKV split", t.DType())
	}
}

func splitTypedFirstAxis3[T float32 | float16.Float16 | bfloat16.BFloat16](t *tensors.Tensor, partDims ...int) ([]*tensors.Tensor, error) {
	flat, err := tensors.CopyFlatData[T](t)
	if err != nil {
		return nil, err
	}
	if len(flat)%3 != 0 {
		return nil, errors.Errorf("unexpected fused QKV element count %d (not divisible by 3)", len(flat))
	}
	partLen := len(flat) / 3
	parts := make([]*tensors.Tensor, 3)
	for idx := range 3 {
		parts[idx] = tensors.FromFlatDataAndDimensions(flat[idx*partLen:(idx+1)*partLen], partDims...)
	}
	return parts, nil
}

// ensureFloat32HostTensor converts Float16, BFloat16, or Float64 host tensors to Float32,
// returning t unchanged if it is already Float32 or integer.
func ensureFloat32HostTensor(t *tensors.Tensor) (*tensors.Tensor, error) {
	dtype := t.DType()
	if dtype == dtypes.Float32 || !dtype.IsFloat() {
		return t, nil
	}
	dims := t.Shape().Dimensions
	out := tensors.FromShape(shapes.Make(dtypes.Float32, dims...))

	switch dtype {
	case dtypes.Float16:
		src, err := tensors.CopyFlatData[float16.Float16](t)
		if err != nil {
			out.FinalizeAll()
			return nil, err
		}
		_ = out.MutableFlatData(func(dstAny any) {
			dst := dstAny.([]float32)
			for i, v := range src {
				dst[i] = v.Float32()
			}
		})
	case dtypes.BFloat16:
		src, err := tensors.CopyFlatData[bfloat16.BFloat16](t)
		if err != nil {
			out.FinalizeAll()
			return nil, err
		}
		_ = out.MutableFlatData(func(dstAny any) {
			dst := dstAny.([]float32)
			for i, v := range src {
				dst[i] = v.Float32()
			}
		})
	case dtypes.Float64:
		src, err := tensors.CopyFlatData[float64](t)
		if err != nil {
			out.FinalizeAll()
			return nil, err
		}
		_ = out.MutableFlatData(func(dstAny any) {
			dst := dstAny.([]float32)
			for i, v := range src {
				dst[i] = float32(v)
			}
		})
	default:
		out.FinalizeAll()
		return t, nil
	}

	t.FinalizeAll()
	return out, nil
}

package transformer_test

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/models/gguf"
	"github.com/gomlx/go-huggingface/models/transformer"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type syntheticGGUFWriter struct {
	buf []byte
}

func (w *syntheticGGUFWriter) u8(v uint8)     { w.buf = append(w.buf, v) }
func (w *syntheticGGUFWriter) u16(v uint16)   { w.buf = binary.LittleEndian.AppendUint16(w.buf, v) }
func (w *syntheticGGUFWriter) u32(v uint32)   { w.buf = binary.LittleEndian.AppendUint32(w.buf, v) }
func (w *syntheticGGUFWriter) u64(v uint64)   { w.buf = binary.LittleEndian.AppendUint64(w.buf, v) }
func (w *syntheticGGUFWriter) f32(v float32)  { w.u32(math.Float32bits(v)) }
func (w *syntheticGGUFWriter) str(s string)   { w.u64(uint64(len(s))); w.buf = append(w.buf, s...) }
func (w *syntheticGGUFWriter) kvStr(k, v string) {
	w.str(k)
	w.u32(8) // GGUF_TYPE_STRING
	w.str(v)
}
func (w *syntheticGGUFWriter) kvU32(k string, v uint32) {
	w.str(k)
	w.u32(4) // GGUF_TYPE_UINT32
	w.u32(v)
}
func (w *syntheticGGUFWriter) kvF32(k string, v float32) {
	w.str(k)
	w.u32(6) // GGUF_TYPE_FLOAT32
	w.f32(v)
}
func (w *syntheticGGUFWriter) kvBool(k string, v bool) {
	w.str(k)
	w.u32(7) // GGUF_TYPE_BOOL
	if v {
		w.u8(1)
	} else {
		w.u8(0)
	}
}
func (w *syntheticGGUFWriter) kvStrArr(k string, vals []string) {
	w.str(k)
	w.u32(9) // GGUF_TYPE_ARRAY
	w.u32(8) // GGUF_TYPE_STRING
	w.u64(uint64(len(vals)))
	for _, s := range vals {
		w.str(s)
	}
}

type syntheticTensorSpec struct {
	name  string
	shape []uint64 // GGUF innermost-first dimensions
	ttype gguf.TensorType
	data  []byte
}

func makeQ4_0Blocks(numBlocks int, scaleBits uint16) []byte {
	out := make([]byte, numBlocks*18)
	for b := range numBlocks {
		base := b * 18
		binary.LittleEndian.PutUint16(out[base:base+2], scaleBits)
		for j := range 16 {
			lo := uint8((b + j) % 16)
			hi := uint8((b + j + 5) % 16)
			out[base+2+j] = lo | (hi << 4)
		}
	}
	return out
}

func makeQ5_0Blocks(numBlocks int, scaleBits uint16) []byte {
	out := make([]byte, numBlocks*22)
	for b := range numBlocks {
		base := b * 22
		binary.LittleEndian.PutUint16(out[base:base+2], scaleBits)
		binary.LittleEndian.PutUint32(out[base+2:base+6], 0xA5A55A5A^uint32(b*17))
		for j := range 16 {
			lo := uint8((b*3 + j) % 16)
			hi := uint8((b*3 + j + 7) % 16)
			out[base+6+j] = lo | (hi << 4)
		}
	}
	return out
}

func makeF32Vec(n int, val float32) []byte {
	out := make([]byte, n*4)
	for i := range n {
		binary.LittleEndian.PutUint32(out[i*4:(i+1)*4], math.Float32bits(val+float32(i%5)*0.01))
	}
	return out
}

func makeF16Mat(n int, valBits uint16) []byte {
	out := make([]byte, n*2)
	for i := range n {
		binary.LittleEndian.PutUint16(out[i*2:(i+1)*2], valBits)
	}
	return out
}

// writeSyntheticBertGGUF writes a self-contained tiny BERT GGUF model (no config.json or tokenizer.json needed).
func writeSyntheticBertGGUF(t *testing.T, filePath string) {
	t.Helper()
	const (
		hidden       = 32
		intermediate = 64
		maxPos       = 16
		vocabSize    = 8
	)
	const scaleHalf = uint16(0x2E66) // ~0.1 in IEEE-754 float16

	specs := []syntheticTensorSpec{
		// Embeddings: Q5_0 token_embd, F16 position_embd, F32 token_types
		{name: "token_embd.weight", shape: []uint64{hidden, vocabSize}, ttype: gguf.TensorTypeQ5_0, data: makeQ5_0Blocks(vocabSize*(hidden/32), scaleHalf)},
		{name: "position_embd.weight", shape: []uint64{hidden, maxPos}, ttype: gguf.TensorTypeF16, data: makeF16Mat(maxPos*hidden, scaleHalf)},
		{name: "token_types.weight", shape: []uint64{hidden, 2}, ttype: gguf.TensorTypeF32, data: makeF32Vec(2*hidden, 0.02)},
		{name: "token_embd_norm.weight", shape: []uint64{hidden}, ttype: gguf.TensorTypeF32, data: makeF32Vec(hidden, 1.0)},
		{name: "token_embd_norm.bias", shape: []uint64{hidden}, ttype: gguf.TensorTypeF32, data: makeF32Vec(hidden, 0.0)},

		// Layer 0: fused QKV in Q4_0 + Q5_0/Q4_0 projections + F32 norms/biases
		{name: "blk.0.attn_qkv.weight", shape: []uint64{hidden, 3 * hidden}, ttype: gguf.TensorTypeQ4_0, data: makeQ4_0Blocks(3*hidden*(hidden/32), scaleHalf)},
		{name: "blk.0.attn_qkv.bias", shape: []uint64{3 * hidden}, ttype: gguf.TensorTypeF32, data: makeF32Vec(3*hidden, 0.01)},
		{name: "blk.0.attn_output.weight", shape: []uint64{hidden, hidden}, ttype: gguf.TensorTypeQ5_0, data: makeQ5_0Blocks(hidden*(hidden/32), scaleHalf)},
		{name: "blk.0.attn_output.bias", shape: []uint64{hidden}, ttype: gguf.TensorTypeF32, data: makeF32Vec(hidden, 0.0)},
		{name: "blk.0.attn_output_norm.weight", shape: []uint64{hidden}, ttype: gguf.TensorTypeF32, data: makeF32Vec(hidden, 1.0)},
		{name: "blk.0.attn_output_norm.bias", shape: []uint64{hidden}, ttype: gguf.TensorTypeF32, data: makeF32Vec(hidden, 0.0)},
		{name: "blk.0.ffn_up.weight", shape: []uint64{hidden, intermediate}, ttype: gguf.TensorTypeQ4_0, data: makeQ4_0Blocks(intermediate*(hidden/32), scaleHalf)},
		{name: "blk.0.ffn_up.bias", shape: []uint64{intermediate}, ttype: gguf.TensorTypeF32, data: makeF32Vec(intermediate, 0.05)},
		{name: "blk.0.ffn_down.weight", shape: []uint64{intermediate, hidden}, ttype: gguf.TensorTypeQ5_0, data: makeQ5_0Blocks(hidden*(intermediate/32), scaleHalf)},
		{name: "blk.0.ffn_down.bias", shape: []uint64{hidden}, ttype: gguf.TensorTypeF32, data: makeF32Vec(hidden, 0.0)},
		{name: "blk.0.layer_output_norm.weight", shape: []uint64{hidden}, ttype: gguf.TensorTypeF32, data: makeF32Vec(hidden, 1.0)},
		{name: "blk.0.layer_output_norm.bias", shape: []uint64{hidden}, ttype: gguf.TensorTypeF32, data: makeF32Vec(hidden, 0.0)},
	}

	w := &syntheticGGUFWriter{}
	w.buf = append(w.buf, "GGUF"...)
	w.u32(3) // version
	w.u64(uint64(len(specs)))
	w.u64(16) // KV count

	w.kvStr("general.architecture", "bert")
	w.kvU32("bert.context_length", maxPos)
	w.kvU32("bert.embedding_length", hidden)
	w.kvU32("bert.block_count", 1)
	w.kvU32("bert.feed_forward_length", intermediate)
	w.kvU32("bert.attention.head_count", 2)
	w.kvF32("bert.attention.layer_norm_epsilon", 1e-12)
	w.kvBool("bert.attention.causal", false)
	w.kvU32("bert.pooling_type", 1) // MEAN pooling

	w.kvStr("tokenizer.ggml.model", "bert")
	w.kvStrArr("tokenizer.ggml.tokens", []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "##s"})
	w.kvU32("tokenizer.ggml.unknown_token_id", 1)
	w.kvU32("tokenizer.ggml.bos_token_id", 2)
	w.kvU32("tokenizer.ggml.seperator_token_id", 3)
	w.kvU32("tokenizer.ggml.padding_token_id", 0)
	w.kvU32("tokenizer.ggml.token_type_count", 2)

	var dataBlob []byte
	for _, s := range specs {
		for len(dataBlob)%32 != 0 {
			dataBlob = append(dataBlob, 0)
		}
		offset := uint64(len(dataBlob))
		dataBlob = append(dataBlob, s.data...)

		w.str(s.name)
		w.u32(uint32(len(s.shape)))
		for _, d := range s.shape {
			w.u64(d)
		}
		w.u32(uint32(s.ttype))
		w.u64(offset)
	}
	for len(w.buf)%32 != 0 {
		w.buf = append(w.buf, 0)
	}
	w.buf = append(w.buf, dataBlob...)
	require.NoError(t, os.WriteFile(filePath, w.buf, 0644))
}

func TestGGUFTransformerEndToEnd(t *testing.T) {
	repoDir := t.TempDir()
	// Include two .gguf files to verify Q4_K_M default selection and WithGGUFFile override.
	q4kmPath := filepath.Join(repoDir, "tiny-bert-Q4_K_M.gguf")
	f16Path := filepath.Join(repoDir, "tiny-bert-f16.gguf")
	writeSyntheticBertGGUF(t, q4kmPath)
	writeSyntheticBertGGUF(t, f16Path)

	repo := hub.NewLocal(repoDir)
	hfModel, err := transformer.LoadModel(repo)
	require.NoError(t, err)

	assert.True(t, hfModel.IsGGUF())
	assert.Equal(t, "tiny-bert-Q4_K_M.gguf", hfModel.GGUFFile())
	assert.False(t, hfModel.UseCausalMask())
	assert.Equal(t, "bert", hfModel.Config.ModelType)
	assert.Equal(t, 32, hfModel.Config.HiddenSize)
	assert.Equal(t, 1, hfModel.Config.NumHiddenLayers)
	require.NotNil(t, hfModel.PoolingConfig)
	assert.True(t, hfModel.PoolingConfig.PoolingModeMeanTokens)
	assert.Len(t, hfModel.Modules, 3)

	// Tokenizer loaded directly from embedded GGUF metadata:
	tok, err := hfModel.GetTokenizer()
	require.NoError(t, err)
	ids := tok.Encode("Hello worlds")
	assert.Equal(t, []int{2, 5, 6, 7, 3}, ids) // [CLS] hello world ##s [SEP]

	// Compare outputs across all official backends (xla:cpu decomposed vs go fused SIMD)
	// for both approximate and exact GELU.
	for _, approxGelu := range []bool{true, false} {
		var refOut [][]float32
		var refFloatOut [][]float32
		testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
			m, err := transformer.LoadModel(repo)
			require.NoError(t, err)
			m.WithApproximateGelu(approxGelu)

			store := model.NewStore()
			require.NoError(t, m.LoadStore(backend, store))

			// Verify quantized variables retain Quantization metadata and Uint8 storage,
			// while F16 position embeddings are converted to Float32.
			tokEmbed := store.GetVariable("/token_embed/embeddings")
			require.NotNil(t, tokEmbed)
			require.NotNil(t, tokEmbed.Quantization())
			assert.Equal(t, compute.GGMLQ5_0, tokEmbed.Quantization().GGMLType)
			assert.Equal(t, dtypes.Uint8, tokEmbed.Shape().DType)

			qWeights := store.GetVariable("/layer_0/attn/MultiHeadAttention/query/dense/weights")
			require.NotNil(t, qWeights)
			require.NotNil(t, qWeights.Quantization())
			assert.Equal(t, compute.GGMLQ4_0, qWeights.Quantization().GGMLType)

			posEmbed := store.GetVariable("/pos_embed/embeddings")
			require.NotNil(t, posEmbed)
			assert.Nil(t, posEmbed.Quantization())
			assert.Equal(t, dtypes.Float32, posEmbed.Shape().DType)

			exec := model.MustNewExec(backend, store, func(s *model.Scope, inputs []*graph.Node) *graph.Node {
				return m.SentenceEmbeddingGraph(s, inputs[0], inputs[1])
			})

			tokenIDs := tensors.FromValue([][]int32{{2, 5, 6, 7, 3, 0}, {2, 5, 3, 0, 0, 0}})
			seqLens := tensors.FromValue([]int32{5, 3})
			outs := exec.MustExec(tokenIDs, seqLens)
			got := outs[0].Value().([][]float32)
			require.Len(t, got, 2)
			require.Len(t, got[0], 32)

			// Embeddings must be L2-normalized.
			for rowIdx, row := range got {
				var normSq float64
				for _, v := range row {
					normSq += float64(v) * float64(v)
				}
				assert.InDeltaf(t, 1.0, math.Sqrt(normSq), 1e-4, "row %d L2 norm", rowIdx)
			}

			if refOut == nil {
				refOut = got
			} else {
				// Note: xla:cpu decomposes QuantizedDense to float32 activations, while go uses
				// GGML Q8_0 activation quantization (matching llama.cpp), yielding <1e-2 max delta
				// and >0.9995 cosine similarity.
				for r := range got {
					var dot float64
					for c := range got[r] {
						dot += float64(refOut[r][c]) * float64(got[r][c])
					}
					assert.Greaterf(t, dot, 0.9995, "approxGelu=%v row %d cosine similarity", approxGelu, r)
					assert.InDeltaSlicef(t, refOut[r], got[r], 1e-2, "approxGelu=%v row %d backend parity", approxGelu, r)
				}
			}

			// With ParamQuantizedMatmul=false (weight-only float32 matmuls), both backends match within 1e-4.
			store.SetParam(layers.ParamQuantizedMatmul, false)
			execFloat := model.MustNewExec(backend, store, func(s *model.Scope, inputs []*graph.Node) *graph.Node {
				return m.SentenceEmbeddingGraph(s, inputs[0], inputs[1])
			})
			outsFloat := execFloat.MustExec(tokenIDs, seqLens)
			gotFloat := outsFloat[0].Value().([][]float32)
			if refFloatOut == nil {
				refFloatOut = gotFloat
			} else {
				for r := range gotFloat {
					assert.InDeltaSlicef(t, refFloatOut[r], gotFloat[r], 1e-4, "approxGelu=%v row %d float32 matmul parity", approxGelu, r)
				}
			}
		})
	}
}

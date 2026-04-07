# **go-huggingface**, download, tokenize and convert models from HuggingFace. 

[![GoDev](https://img.shields.io/badge/go.dev-reference-007d9c?logo=go&logoColor=white)](https://pkg.go.dev/github.com/gomlx/go-huggingface?tab=doc)
[![Tests](https://github.com/gomlx/go-huggingface/actions/workflows/linux_amd64_tests.yaml/badge.svg)](https://github.com/gomlx/go-huggingface/actions/workflows/linux_amd64_tests.yaml)
[![Slack](https://img.shields.io/badge/Slack-GoMLX-purple.svg?logo=slack)](https://app.slack.com/client/T029RQSE6/C08TX33BX6U)
[![Sponsor gomlx](https://img.shields.io/badge/Sponsor-gomlx-white?logo=github&style=flat-square)](https://github.com/sponsors/gomlx)

## 📖 Overview

Simple APIs for downloading (`hub`), tokenizing (`tokenizers`), (**experimental**) model conversion (`models/transformers`) of 
[HuggingFace🤗](huggingface.co) transformer models using [GoMLX](https://github.com/gomlx/gomlx), and last but not least, simplified datasets (parquet based) downloading and scanning.

Each component is independent, and only depends on what it needs -- `hub` has no dependency on `GoMLX`, `tokenizers` has no dependence on `parquet-go` (to parse datasets), etc.

It also provides a `bucket` library to bucketize sentences to be tokenized into buckets of increasing sizes (e.g.: powers-of-2, two-bits, etc.) with automatic padding, and
maximum delay configuration (for online systems).

See examples:
 
* [MS MARCO dataset](https://github.com/gomlx/go-huggingface/tree/main/examples/msmarco): 
  a small library that makes it easy access to this specific dataset, and serves as an example to access others.
* [Tencent's KaLM-Embedding-Gemma3-12B-2511 Sentence Encoder](https://github.com/gomlx/go-huggingface/tree/main/examples/kalmgemma3): 
  a small library that makes it trivial to use this model and serves as an example how to use others.


🚧 **EXPERIMENTAL and IN DEVELOPMENT**: By no means it covers all models/tokenizers/dataset types in HuggingFace, but support is continuously expanding (we add support for the models we are using, or when someone asks for). Models are easy to run, datasets are easy to scan, tokenizers come configured from HuggingFace, etc. But ... it is still under development -- and on that note: contributions and suggestions are most welcome.

## Packages `hub`: Downloding info and files from a repository

### Preamble: Imports And Variables

```go
import (
    "github.com/gomlx/go-huggingface/hub"
    "github.com/gomlx/go-huggingface/tokenizers"
)

var (
	// HuggingFace authentication token read from environment.
	// It can be created in https://huggingface.co
	// Some files may require it for downloading.
	hfAuthToken = os.Getenv("HF_TOKEN")

	// Model IDs we use for testing.
	hfModelIDs = []string{
		"google/gemma-2-2b-it",
		"sentence-transformers/all-MiniLM-L6-v2",
		"protectai/deberta-v3-base-zeroshot-v1-onnx",
		"KnightsAnalytics/distilbert-base-uncased-finetuned-sst-2-english",
		"KnightsAnalytics/distilbert-NER",
		"SamLowe/roberta-base-go_emotions-onnx",
	}
)
```

### List files for each model

```go
for _, modelID := range hfModelIDs {
	fmt.Printf("\n%s:\n", modelID)
	repo := hub.New(modelID).WithAuth(hfAuthToken)
	for fileName, err := range repo.IterFileNames() {
		if err != nil { panic(err) }
		fmt.Printf("\t%s\n", fileName)
	}
}
```

The result looks like this:

```
google/gemma-2-2b-it:
	.gitattributes
	README.md
	config.json
	generation_config.json
	model-00001-of-00002.safetensors
	model-00002-of-00002.safetensors
	model.safetensors.index.json
	special_tokens_map.json
	tokenizer.json
	tokenizer.model
	tokenizer_config.json
…
```

### List tokenizer classes for each model

```go
for _, modelID := range hfModelIDs {
	fmt.Printf("\n%s:\n", modelID)
	repo := hub.New(modelID).WithAuth(hfAuthToken)
	config, err := tokenizers.GetConfig(repo)
	if err != nil { panic(err) }
	fmt.Printf("\ttokenizer_class=%s\n", config.TokenizerClass)
}
```

Results:

```
google/gemma-2-2b-it:
	tokenizer_class=GemmaTokenizer

sentence-transformers/all-MiniLM-L6-v2:
	tokenizer_class=BertTokenizer

protectai/deberta-v3-base-zeroshot-v1-onnx:
	tokenizer_class=DebertaV2Tokenizer
…
```

## Package `tokenizers`: an API and a set of tokenizer implementations

### Tokenize for using Go-only "SentencePiece" tokenizer (for all Gemma models)

* The output "Downloaded" message happens only the tokenizer file is not yet cached, so only the first time:

```go
repo := hub.New("google/gemma-2-2b-it").WithAuth(hfAuthToken)
tokenizer, err := tokenizers.New(repo)
if err != nil { panic(err) }

sentence := "The book is on the table."
tokens := tokenizer.Encode(sentence)
fmt.Printf("Sentence:\t%s\n", sentence)
fmt.Printf("Tokens:  \t%v\n", tokens)
```

```
Downloaded 1/1 files, 4.2 MB downloaded         
Sentence:	The book is on the table.
Tokens:  	[651 2870 603 611 573 3037 235265]
```

### Tokenize and "Bucketize" sentences (using "two-bits" bucketing strategy)

The library also provides the `github.com/gomlx/go-huggingface/tokenizers/bucket` package to
bucket sentences in similar length ones, which can then be used to create batches of tokens
with minimal padding.

If provides different bucketing strategies (e.g.: Power-of-2, Power-of-X, Two-Bits, etc.), 
and maximum latency waiting for buckets (for online usage), parallelization of tokenization,
and is very simple to use:

Example:
* Write individual sentences to `bucketsInputChan`.
* Read "batched" buckets from `bucketsOutputChan`.
* Close `bucketsInputChan` when done, it will automatically close
  `bucketsOutputChan` once all the buffers are drained.
* Wait for `wg` to finish.

```go
tokenizer := ... // see previous example

// Start bucket runner in a separate goroutine.
var wg sync.WaitGroup
bucketsInputChan := make(chan bucket.SentenceRef)
bucketsOutputChan := make(chan bucket.Bucket, 10)
bkt := bucket.New(tokenizer).
	ByTwoBitBucketBudget(8*1024, 16).  // ~8K total tokens per bucket, ~20% padding overhead
	WithMaxParallelization(-1)
wg.Go(func() {
	bkt.Run(bucketsInputChan, bucketsOutputChan)
})
...
```

### Tokenize for a [Sentence Transformer](https://www.sbert.net/) derived model, using Rust's based [github.com/daulet/tokenizers](https://github.com/daulet/tokenizers) tokenizer

For most tokenizers in HuggingFace though, there is no Go-only version yet, and for now we use the 
[github.com/daulet/tokenizers](https://github.com/daulet/tokenizers), which is based on a fast tokenizer written in Rust.

It requires installation of the built Rust library though, 
see [github.com/daulet/tokenizers](https://github.com/daulet/tokenizers) on how to install it, 
they provide prebuilt binaries.

> **Note**: `daulet/tokenizers` also provides a simple downloader, so `go-huggingface` is not strictly necessary -- 
> if you don't want the extra dependency and only need the tokenizer, you don't need to use it. `go-huggingface` 
> helps by allowing also downloading other files (models, datasets), and a shared cache across different projects 
> and `huggingface-hub` (the python downloader library).

```go
import dtok "github.com/daulet/tokenizers"

%%
modelID := "KnightsAnalytics/all-MiniLM-L6-v2"
repo := hub.New(modelID).WithAuth(hfAuthToken)
localFile := must.M1(repo.DownloadFile("tokenizer.json"))
tokenizer := must.M1(dtok.FromFile(localFile))
defer tokenizer.Close()
tokens, _ := tokenizer.Encode(sentence, true)

fmt.Printf("Sentence:\t%s\n", sentence)
fmt.Printf("Tokens:  \t%v\n", tokens)
```

```
Sentence:	The book is on the table.
Tokens:  	[101 1996 2338 2003 2006 1996 2795 1012 102 0 0 0…]
```

## Package [`onnx-gomlx`](https://github.com/gomlx/onnx-gomlx): convert ONNX models to GoMLX

### Download and execute ONNX model for [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

Only the first 3 lines are actually demoing `go-huggingface`.
The remainder lines uses [`github.com/gomlx/onnx-gomlx`](https://github.com/gomlx/onnx-gomlx)
to parse and convert the ONNX model to GoMLX, and then
[`github.com/gomlx/gomlx`](github.com/gomlx/gomlx) to execute the converted model
for a couple of sentences.

```go
// Get ONNX model.
repo := hub.New("sentence-transformers/all-MiniLM-L6-v2").WithAuth(hfAuthToken)
onnxFilePath, err := repo.DownloadFile("onnx/model.onnx")
if err != nil { panic(err) }
onnxModel, err := onnx.ReadFile(onnxFilePath)
if err != nil { panic(err) }

// Convert ONNX variables to GoMLX context (which stores variables):
ctx := context.New()
err = onnxModel.VariablesToContext(ctx)
if err != nil { panic(err) }

// Test input.
sentences := []string{
	"This is an example sentence",
	"Each sentence is converted"}
inputIDs := [][]int64{
	{101, 2023, 2003, 2019, 2742, 6251,  102},
	{ 101, 2169, 6251, 2003, 4991,  102,    0}}
tokenTypeIDs := [][]int64{
	{0, 0, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0}}
attentionMask := [][]int64{
	{1, 1, 1, 1, 1, 1, 1},
	{1, 1, 1, 1, 1, 1, 0}}

// Execute GoMLX graph with model.
embeddings := context.ExecOnce(
	backends.New(), ctx,
	func (ctx *context.Context, inputs []*graph.Node) *graph.Node {
		modelOutputs := onnxModel.CallGraph(ctx, inputs[0].Graph(), map[string]*graph.Node{
			"input_ids": inputs[0],
			"attention_mask": inputs[1],
			"token_type_ids": inputs[2]})
		return modelOutputs[0]
	}, 
	inputIDs, attentionMask, tokenTypeIDs)

fmt.Printf("Sentences: \t%q\n", sentences)
fmt.Printf("Embeddings:\t%s\n", embeddings)
```

```
Sentences: 	["This is an example sentence" "Each sentence is converted"]
Embeddings:	[2][7][384]float32{
 {{0.0366, -0.0162, 0.1682, ..., 0.0554, -0.1644, -0.2967},
  {0.7239, 0.6399, 0.1888, ..., 0.5946, 0.6206, 0.4897},
  {0.0064, 0.0203, 0.0448, ..., 0.3464, 1.3170, -0.1670},
  ...,
  {0.1479, -0.0643, 0.1457, ..., 0.8837, -0.3316, 0.2975},
  {0.5212, 0.6563, 0.5607, ..., -0.0399, 0.0412, -1.4036},
  {1.0824, 0.7140, 0.3986, ..., -0.2301, 0.3243, -1.0313}},
 {{0.2802, 0.1165, -0.0418, ..., 0.2711, -0.1685, -0.2961},
  {0.8729, 0.4545, -0.1091, ..., 0.1365, 0.4580, -0.2042},
  {0.4752, 0.5731, 0.6304, ..., 0.6526, 0.5612, -1.3268},
  ...,
  {0.6113, 0.7920, -0.4685, ..., 0.0854, 1.0592, -0.2983},
  {0.4115, 1.0946, 0.2385, ..., 0.8984, 0.3684, -0.7333},
  {0.1374, 0.5555, 0.2678, ..., 0.5426, 0.4665, -0.5284}}}
```

## Package `models/transformers`: import HuggingFace transformer models as GoMLX ones

> **EXPERIMENTAL**: fresh from the oven, and likely only works for few models now, but it should be easy to extend the support for other models.

The `models/transformer` package allows downloading and inspecting HuggingFace transformer models, reading their configurations and weights, and building a `GoMLX` computation graph dynamically based on the model architectures (such as `sentence_transformers` pipelines).

### Example with `tencent/KaLM-Embedding-Gemma3-12B-2511`

```go
import (
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/models/transformer"
	"github.com/gomlx/gomlx/pkg/ml/context"
)

// 1. Download configuration and weights from HuggingFace
repo := hub.New("tencent/KaLM-Embedding-Gemma3-12B-2511").WithAuth(hfAuthToken)
model, err := transformer.LoadModel(repo)
if err != nil { panic(err) }

// Print a summary of the model features and sizes:
fmt.Println(model.Description())

// 2. Load the loaded weights to a GoMLX context
ctx := context.New()
model.LoadContext(ctx)

// 3. Build a GoMLX graph for the model. 
// Assuming `inputTokens` is a `*graph.Node` with shape [batch_size, sequence_length]
// embeddings := model.BuildGraph(ctx, inputTokens)
```

## Package `datasets`: download info, files or iterate directly over Parquet records of HuggingFace datasets

The `datasets` package provides functionality to retrieve dataset information and download files, integrated with `hub`. We are going to use the [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) as an example, exploring its structure and downloading one of its sample files (~2.5Gb of data) to parse the `.parquet` file.

First, you can use the `datasets` package to understand the dataset structure:

```go
import "github.com/gomlx/go-huggingface/datasets"

// Print dataset info: configurations, splits, sizes and features.
ds := datasets.New("HuggingFaceFW/fineweb").WithAuth(hfAuthToken)
fmt.Println(ds.String())
```

### Structure of file

You can auto-generate the Go struct for the dataset using the `generate_dataset_structs` command line tool:

```bash
go run github.com/gomlx/go-huggingface/cmd/generate_dataset_structs -dataset HuggingFaceFW/fineweb -config sample-10BT
```

Result:

```go
var (
    FineWebID = "HuggingFaceFW/fineweb"
    FineWebSampleFile = "sample/10BT/000_00000.parquet"
)

// FinewebRecord was auto-generated by cmd/generate_dataset_structs.
// The parquet annotations are described in: https://pkg.go.dev/github.com/parquet-go/parquet-go#SchemaOf
type FinewebRecord struct {
	Date          string  `json:"date" parquet:"date"`
	Dump          string  `json:"dump" parquet:"dump"`
	FilePath      string  `json:"file_path" parquet:"file_path"`
	ID            string  `json:"id" parquet:"id"`
	Language      string  `json:"language" parquet:"language"`
	LanguageScore float64 `json:"language_score" parquet:"language_score"`
	Text          string  `json:"text" parquet:"text,snappy"`
	TokenCount    int64   `json:"token_count" parquet:"token_count"`
	URL           string  `json:"url" parquet:"url,snappy"`
}
```

Now we can read the `parquet` files into the `FinewebRecord` records:

```go
import (
    "fmt"
    "github.com/gomlx/go-huggingface/datasets"
)

func main() {
    // Initialize the dataset reference.
    ds := datasets.New(FineWebID).WithAuth(hfAuthToken)
    
    // Iterate over all records in the dataset:
    // Warning: for FineWeb this will download the entire 15TB dataset. 
    // You can break early, but the initial download request might still be large.
    // For manual samples, you can also use datasets.IterParquetFromFile(localFile).
    ii := 0
    for row, err := range datasets.IterParquetFromDataset[FinewebRecord](ds, "sample-10BT", "train") {
        if err != nil {
            panic(err)
        }
        fmt.Printf("Row %0d:\tScore=%.3f Text=[%q], URL=[%s]\n", ii, row.LanguageScore, TrimString(row.Text, 50), TrimString(row.URL, 40))
        ii++
        if ii >= 10 {
            break
        }
    }
    fmt.Printf("%d rows read\n", ii)
}

// TrimString returns s trimmed to at most maxLength runes. If trimmed it appends "…" at the end.
func TrimString(s string, maxLength int) string {
    if utf8.RuneCountInString(s) <= maxLength {
        return s
    }
    runes := []rune(s)
    return string(runes[:maxLength-1]) + "…"
}
```

Results:

```
10 rows read
Row 0:	Score=0.823 Text=["|Viewing Single Post From: Spoilers for the Week …"], URL=[http://daytimeroyaltyonline.com/single/…]
Row 1:	Score=0.974 Text=["*sigh* Fundamentalist community, let me pass on s…"], URL=[http://endogenousretrovirus.blogspot.co…]
Row 2:	Score=0.873 Text=["A novel two-step immunotherapy approach has shown…"], URL=[http://news.cancerconnect.com/]
Row 3:	Score=0.932 Text=["Free the Cans! Working Together to Reduce Waste\nI…"], URL=[http://sharingsolution.com/2009/05/23/f…]
…
```

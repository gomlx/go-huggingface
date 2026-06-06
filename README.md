# **go-huggingface**, download, tokenize and convert models from HuggingFace. 

[![GoDev](https://img.shields.io/badge/go.dev-reference-007d9c?logo=go&logoColor=white)](https://pkg.go.dev/github.com/gomlx/go-huggingface?tab=doc)
[![Tests](https://github.com/gomlx/go-huggingface/actions/workflows/linux_amd64_tests.yaml/badge.svg)](https://github.com/gomlx/go-huggingface/actions/workflows/linux_amd64_tests.yaml)
[![Slack](https://img.shields.io/badge/Slack-GoMLX-purple.svg?logo=slack)](https://app.slack.com/client/T029RQSE6/C08TX33BX6U)
[![Sponsor gomlx](https://img.shields.io/badge/Sponsor-gomlx-white?logo=github&style=flat-square)](https://github.com/sponsors/gomlx)

## 📖 Overview

Simple APIs for **downloading** (`hub`), **tokenizing** (`tokenizers`), (**experimental**) **model conversion** (`models/transformer`) of 
[HuggingFace🤗](https://huggingface.co) transformer models using [GoMLX](https://github.com/gomlx/gomlx), and last but not least, **datasets (parquet-based) downloading and scanning**.

Each component is independent, and only depends on what it needs -- `hub` has no dependency on `GoMLX`, `tokenizers` has no dependency on `parquet-go` (to parse datasets), etc.

It also provides a `bucket` library to bucketize sentences to be tokenized into buckets of increasing sizes (e.g.: powers-of-2, two-bits, etc.) with automatic padding, and
maximum delay configuration (for online systems).

See examples:
 
* [MS MARCO dataset](https://github.com/gomlx/go-huggingface/tree/main/examples/msmarco): 
  a small library that provides easy access to this specific dataset, and serves as an example of how to access others.
  It includes [benchmark_embed](https://github.com/gomlx/go-huggingface/tree/main/examples/msmarco/benchmark_embed/),
  a command-line benchmark of sentence embeddings, that also serves as an example of how to use the library.
* [Tencent's KaLM-Embedding-Gemma3-12B-2511 Sentence Encoder](https://github.com/gomlx/go-huggingface/tree/main/examples/kalmgemma3): 
  a small library that makes it trivial to use this model and serves as an example of how to use others.
* [BAAI (Beijing Academy of Artificial Intelligence) BGE Small Sentence Embedder (English) v1.5](https://github.com/gomlx/go-huggingface/tree/main/examples/BAAI-bge-small-en-v1.5): a small and very performant sentence embedder (BERT-based).
  
🚧 **EXPERIMENTAL and IN DEVELOPMENT**: By no means does it cover all models/tokenizers/dataset types in HuggingFace, but support is continuously expanding (we add support for the models we are using, or when someone asks for it). Models are easy to run, datasets are easy to scan, tokenizers come configured from HuggingFace, etc. But ... it is still under development -- and on that note: contributions and suggestions are most welcome.



---

## Info/Download from HuggingFace Hub

**Package**: `github.com/gomlx/go-huggingface/hub`

It provides information from any repo in the Hub (models, datasets, etc.), and provides a very simple
API to download files, sharing the cache format with the original HuggingFace library (so both share the same cache).

### Preamble: Imports And Variables

```go
import (
    "fmt"
    "os"

    "github.com/gomlx/compute/support/humanize"
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
	for fileInfo, err := range repo.IterFileInfos() {
		if err != nil { panic(err) }
		fmt.Printf("\t%s - %s\n", fileInfo.Name, humanize.Bytes(fileInfo.Size))
	}
}
```

The result looks like this:

```
google/gemma-2-2b-it:
	.gitattributes - 1.5 KiB
	README.md - 28.4 KiB
	config.json - 838 B
	generation_config.json - 187 B
	model-00001-of-00002.safetensors - 4.6 GiB
	model-00002-of-00002.safetensors - 229.5 MiB
	model.safetensors.index.json - 23.7 KiB
	special_tokens_map.json - 636 B
	tokenizer.json - 16.7 MiB
	tokenizer.model - 4.0 MiB
	tokenizer_config.json - 45.9 KiB
…
```


---

## HuggingFace Tokenizers

**Package**: `github.com/gomlx/go-huggingface/tokenizers`

The `tokenizers` package provides a generic `Tokenizer` API and a set of tokenizer implementations.

### Tokenize using the Go-only "SentencePiece" tokenizer (for all Gemma models)

* The output "Downloaded" message happens only when the tokenizer file is not yet cached, so only the first time:

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


### Tokenize and "Bucketize" sentences (using "two-bits" bucketing strategy)

The library also provides the `github.com/gomlx/go-huggingface/tokenizers/bucket` package to
bucket sentences in similar length ones, which can then be used to create batches of tokens
with minimal padding.

It provides different bucketing strategies (e.g.: Power-of-2, Power-of-X, Two-Bits, etc.), 
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

### Tokenize for a [Sentence Transformer](https://www.sbert.net/) derived model, using the Rust-based [github.com/daulet/tokenizers](https://github.com/daulet/tokenizers) package

If you don't find a Go tokenizer, or if you need the most performant tokenizer (usually tokenization is not a bottleneck), you can also use the 
[github.com/daulet/tokenizers](https://github.com/daulet/tokenizers) package, which is based on a fast tokenizer written in Rust.

It requires installation of the built Rust library though, 
see [github.com/daulet/tokenizers](https://github.com/daulet/tokenizers) on how to install it, 
they provide prebuilt binaries.

> **Note**: `daulet/tokenizers` also provides a simple downloader, so `go-huggingface` is not strictly necessary -- 
> if you don't want the extra dependency and only need the tokenizer, you don't need to use it. `go-huggingface` 
> helps by also allowing the download of other files (models, datasets), and sharing the cache across different projects 
> with `huggingface-hub` (the Python downloader library).

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



---

## Importing HuggingFace Transformer Models in GoMLX

**Package**: `github.com/gomlx/go-huggingface/models/transformer`

> 🚧 **EXPERIMENTAL**: fresh from the oven, and likely only works for a few models now, but it should be easy to extend the support for other models -- feel free add an issue to any model you want to use.

The `models/transformer` package allows downloading and inspecting HuggingFace transformer models, reading their configurations and weights, and building a `GoMLX` computation graph dynamically based on the model architectures (such as `sentence_transformers` pipelines).

### Example with `tencent/KaLM-Embedding-Gemma3-12B-2511`

See full example in [`./examples/kalmgemma3/kalmgemma3_test.go`](https://github.com/gomlx/go-huggingface/blob/main/examples/kalmgemma3/kalmgemma3_test.go)

```go
import (
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/models/transformer"
	"github.com/gomlx/gomlx/ml/model"
)

// 1. Download configuration and weights from HuggingFace
repo := hub.New("tencent/KaLM-Embedding-Gemma3-12B-2511").WithAuth(hfAuthToken)
hfModel, err := transformer.LoadModel(repo)
if err != nil { panic(err) }
hfModel.WithCausalMask(true)
tokenizer := must1(hfModel.GetTokenizer())
padID, err := tokenizer.SpecialTokenID(api.TokPad)

// Print a summary of the model features and sizes:
fmt.Println(hfModel.Description())

// 2. Load the loaded weights to a GoMLX context
backend := compute.MustNew()
store := model.NewStore()
err := hfModel.LoadStore(backend, store)  // Load model weights into the store.

// 3. Build a GoMLX graph and executor for the model.
kalmExec, err := model.NewExec1(testBackend, testStore, func(scope *model.Scope, tokens *graph.Node) *graph.Node {
	x := hfModel.SentenceEmbeddingGraph(scope, tokens, nil)
	return graph.ConvertDType(x, dtypes.Float32)
})

// 4. Embed sentences ...
tokens := tokenzier.Encode(prompt)
embeddings := kalmExec.MustCall(tokens)

//...
```



---

## HuggingFace Datasets

**Package**: `github.com/gomlx/go-huggingface/datasets`

The `datasets` package provides functionality to retrieve dataset information, download files and iterate over
individual records in a performant way.

We are going to use [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) as an example,
exploring its structure and downloading one of its sample files (~2.0 GiB of data) to parse the `.parquet` file.

First, you can use the `datasets` package to understand the dataset structure:

```go
import "github.com/gomlx/go-huggingface/datasets"

// Print dataset info: configurations, splits, sizes and features.
ds := datasets.New("HuggingFaceFW/fineweb").WithAuth(hfAuthToken)
fmt.Println(ds.String())
```

Outputs:

```
Dataset ID: HuggingFaceFW/fineweb
... (a large list of historical versions) ...
Config: default
  Features: date, dump, file_path, id, language, language_score, text, token_count, url
  Splits: train (25.9G records, 85.7 TiB, 675 files)

Config: sample-100BT
  Features: date, dump, file_path, id, language, language_score, text, token_count, url
  Splits: train (147.6M records, 438.4 GiB)

Config: sample-10BT
  Features: date, dump, file_path, id, language, language_score, text, token_count, url
  Splits: train (14.9M records, 46.7 GiB)

Config: sample-350BT
  Features: date, dump, file_path, id, language, language_score, text, token_count, url
  Splits: train (518.5M records, 1.4 TiB)
```

### Parquet Structure 

We use the `github.com/gomlx/go-huggingface/cmd/generate_dataset_structs` to generate the Go 
structure for the Parquet files:

```bash
go run ./cmd/generate_dataset_structs -dataset HuggingFaceFW/fineweb
```

Outputs:

```go
//... (some info messages)...
type FinewebRecord struct {
	Text string `json:"text" parquet:"text"`
	ID string `json:"id" parquet:"id"`
	Dump string `json:"dump" parquet:"dump"`
	URL string `json:"url" parquet:"url"`
	Date string `json:"date" parquet:"date"`
	FilePath string `json:"file_path" parquet:"file_path"`
	Language string `json:"language" parquet:"language"`
	LanguageScore float64 `json:"language_score" parquet:"language_score"`
	TokenCount int64 `json:"token_count" parquet:"token_count"`
}
```

### Iterating (Reading) Over Parquet Files

With the `struct FinewebRecord` created, we can now iterate over the parquet files. The `IterParquetFromDataset` will 
iterate over all parquet records if allowed, but it will only download one file at a time, so since we only list 10 
rows, it will only download the first file (about 2.2Gb).

```go
var (
    FineWebID = "HuggingFaceFW/fineweb"
    FineWebConfig = "sample-10BT"
    FineWebSplit = "train"
)

// FinewwebRecord generated with github.com/gomlx/go-huggingface/cmd/generate_dataset_structs
type FinewebRecord struct {
	Text string `json:"text" parquet:"text"`
	ID string `json:"id" parquet:"id"`
	Dump string `json:"dump" parquet:"dump"`
	URL string `json:"url" parquet:"url"`
	Date string `json:"date" parquet:"date"`
	FilePath string `json:"file_path" parquet:"file_path"`
	Language string `json:"language" parquet:"language"`
	LanguageScore float64 `json:"language_score" parquet:"language_score"`
	TokenCount int64 `json:"token_count" parquet:"token_count"`
}

// TrimString returns s trimmed to at most maxLength runes. If trimmed it appends "…" at the end.
func TrimString(s string, maxLength int) string {
    if utf8.RuneCountInString(s) <= maxLength {
        return s
    }
    runes := []rune(s)
    return string(runes[:maxLength-1]) + "…"
}

%%
ds := datasets.New(FineWebID)
ds.Verbosity = 2
count := 0
const limit = 10
for row, err := range datasets.IterParquetFromDataset[FinewebRecord](ds, FineWebConfig, FineWebSplit) {
    if err != nil { panic(err) }
	fmt.Printf("Record #%02d:\tScore=%.3f Text=%q, URL=[%s]\n", count+1, row.LanguageScore, TrimString(row.Text, 50), TrimString(row.URL, 40))
    count++
    if count >= limit { break }
}
fmt.Printf("%d records read!", count)
```

Outputs:

```
Record #01:	Score=0.823 Text="|Viewing Single Post From: Spoilers for the Week …", URL=[http://daytimeroyaltyonline.com/single/…]
Record #02:	Score=0.974 Text="*sigh* Fundamentalist community, let me pass on s…", URL=[http://endogenousretrovirus.blogspot.co…]
Record #03:	Score=0.873 Text="A novel two-step immunotherapy approach has shown…", URL=[http://news.cancerconnect.com/]
Record #04:	Score=0.932 Text="Free the Cans! Working Together to Reduce Waste\nI…", URL=[http://sharingsolution.com/2009/05/23/f…]
Record #05:	Score=0.955 Text="ORLANDO, Fla. — While the Rapid Recall Exchange, …", URL=[http://supermarketnews.com/food-safety/…]
Record #06:	Score=0.954 Text="September 28, 2010\n2010 Season - Bowman pulls dow…", URL=[http://www.augustana.edu/x22236.xml]
Record #07:	Score=0.967 Text="Kraft Foods has taken the Cadbury chocolate brand…", URL=[http://www.fdin.org.uk/2012/01/kraft-la…]
Record #08:	Score=0.874 Text="You must be a registered member to view this page…", URL=[http://www.golivewire.com/forums/profil…]
Record #09:	Score=0.912 Text="|Facility Type:||Full Service Restaurant|\n|Inspec…", URL=[http://www.healthspace.com/Clients/VDH/…]
Record #10:	Score=0.925 Text="News of the Week\nBarrie Spring Studio Tour\nApril …", URL=[http://www.jillpricestudios.ca/artist/w…]
10 records read!
…
```



---

## HuggingFace ONNX models

**Package**: `github.com/gomlx/onnx-gomlx/onnx/`

The [ONNX-GoMLX project](https://github.com/gomlx/onnx-gomlx) can convert ONNX models to GoMLX.
It can be used for simple inference, fine-tuning, combining models, etc.
It can even export updated-weights back to the ONNX model.

The example below reads the `.onnx` model using a repo created with the package `hub`, creates a `tokenizer`,
uses the `bucket` to package a list of sentences into a padded batch, converts the model to a GoMLX model
and then executes it on the batch.


**Model**:  ONNX model for [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

Only the first 3 lines are actually demoing `go-huggingface`.
The remainder lines uses [`github.com/gomlx/onnx-gomlx`](https://github.com/gomlx/onnx-gomlx)
to parse and convert the ONNX model to GoMLX, and then
[`github.com/gomlx/gomlx`](github.com/gomlx/gomlx) to execute the converted model
for a couple of sentences.

```go
import (
    "github.com/gomlx/compute"
    "github.com/gomlx/go-huggingface/tokenizers/bucket"
    onnxparser "github.com/gomlx/onnx-gomlx/onnx/parser"
    "github.com/gomlx/gomlx/core/graph"
    "github.com/gomlx/gomlx/ml/model"

    // Default backends.
    _ "github.com/gomlx/gomlx/backends/default"
)

%%
// Get ONNX model.
repo := hub.New("sentence-transformers/all-MiniLM-L6-v2").WithAuth(hfAuthToken)
onnxFilePath, err := repo.DownloadFile("onnx/model.onnx")
if err != nil { panic(err) }
onnxModel, err := onnxparser.FromFile(onnxFilePath)
if err != nil { panic(err) }

// Convert ONNX variables to a GoMLX store:
store := model.NewStore()
err = onnxModel.VariablesToScope(store.RootScope())
if err != nil { panic(err) }

// Tokenize sentences.
tokenizer := must.M1(tokenizers.New(repo))
sentences := []string{
    "This is an example sentence", 
    "Each sentence is converted"}
batchSize := len(sentences)
sentencesChan := make(chan bucket.SentenceRef, batchSize)
bucketChan := make(chan bucket.Bucket, 1)
bucketizer := bucket.New(tokenizer).ByTwoBitBucket(batchSize, 8)
var wg sync.WaitGroup
wg.Go(func() { bucketizer.Run(sentencesChan, bucketChan) })
wg.Go(func() { 
    for i, s := range sentences { sentencesChan <- bucket.SentenceRef{s, i} }
    close(sentencesChan)
})

// Create GoMLX model, and its executor:
miniLMExec := model.MustNewExec1(
    compute.MustNew(), store, 
    func (scope *model.Scope, tokenIDs *graph.Node) *graph.Node {
        tokenIDs = graph.Reshape(tokenIDs, batchSize, -1)
        mask := graph.LogicalNot(graph.IsZero(tokenIDs))
        return onnxModel.CallGraph(scope, tokenIDs.Graph(), map[string]*graph.Node{
            "input_ids": tokenIDs,
            "attention_mask": graph.ConvertDType(mask, dtypes.Int64),
            "token_type_ids": graph.ZerosLike(tokenIDs)})[0]
    })

// Loop over batches:
for bucket := range bucketChan {
    tokenIDs := bucket.Batch
    embeddings := miniLMExec.MustCall(tokenIDs)
    fmt.Printf("Tokens: \t%v\n", tokenIDs)
    fmt.Printf("Embeddings:\t%s\n", embeddings)
}
```

Output:

```
Tokens: 	[101 2023 2003 2019 2742 6251 102 0 101 2169 6251 2003 4991 102 0 0]
Embeddings:	[2][8][384]float32{
 {{0.03652, -0.01617, 0.1683, ..., 0.05541, -0.1644, -0.2968},
  {0.7242, 0.6394, 0.189, ..., 0.5943, 0.6209, 0.4898},
  {0.006568, 0.02115, 0.04448, ..., 0.3471, 1.318, -0.1673},
  ...,
  {0.5212, 0.6566, 0.561, ..., -0.03989, 0.04128, -1.404},
  {1.083, 0.714, 0.3987, ..., -0.2289, 0.3248, -1.031},
  {-0.1745, 0.1791, 0.5735, ..., 0.1578, 0.002306, -0.4539}},
 {{0.2801, 0.1163, -0.04202, ..., 0.271, -0.1684, -0.2962},
  {0.8735, 0.4541, -0.1089, ..., 0.1362, 0.4584, -0.2045},
  {0.475, 0.5726, 0.6299, ..., 0.6521, 0.5611, -1.327},
  ...,
  {0.4114, 1.094, 0.2384, ..., 0.8982, 0.3684, -0.7336},
  {0.1354, 0.5587, 0.2699, ..., 0.5424, 0.47, -0.5306},
  {0.2323, 0.2985, 0.1732, ..., 0.4245, 0.07187, -0.3455}}}
  ```

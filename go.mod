module github.com/gomlx/go-huggingface

go 1.25.0

require (
	github.com/dustin/go-humanize v1.0.1
	github.com/eliben/go-sentencepiece v0.7.0
	github.com/gofrs/flock v0.13.0
	github.com/gomlx/gomlx v0.27.3-0.20260402105318-bf1606661086
	github.com/google/uuid v1.6.0
	github.com/parquet-go/parquet-go v0.29.0
	github.com/pkg/errors v0.9.1
	github.com/stretchr/testify v1.11.1
	github.com/x448/float16 v0.8.4
	golang.org/x/text v0.35.0
	google.golang.org/protobuf v1.36.11
	k8s.io/klog/v2 v2.140.0
)

require (
	github.com/andybalholm/brotli v1.2.0 // indirect
	github.com/davecgh/go-spew v1.1.2-0.20180830191138-d8f796af33cc // indirect
	github.com/go-logr/logr v1.4.3 // indirect
	github.com/gomlx/go-xla v0.2.2 // indirect
	github.com/klauspost/compress v1.18.5 // indirect
	github.com/parquet-go/bitpack v1.0.0 // indirect
	github.com/parquet-go/jsonlite v1.5.0 // indirect
	github.com/pierrec/lz4/v4 v4.1.26 // indirect
	github.com/pmezard/go-difflib v1.0.1-0.20181226105442-5d4384ee4fb2 // indirect
	github.com/twpayne/go-geom v1.6.1 // indirect
	golang.org/x/exp v0.0.0-20260312153236-7ab1446f8b90 // indirect
	golang.org/x/sys v0.42.0 // indirect
	golang.org/x/term v0.40.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)

tool github.com/gomlx/go-huggingface/cmd/generate_dataset_structs

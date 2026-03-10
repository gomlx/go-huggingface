module github.com/gomlx/go-huggingface

go 1.26

require (
	github.com/dustin/go-humanize v1.0.1
	github.com/eliben/go-sentencepiece v0.7.0
	github.com/gofrs/flock v0.13.0
	github.com/gomlx/gomlx v0.26.1-0.20260114072028-dd1b582c66f7
	github.com/google/uuid v1.6.0
	github.com/pkg/errors v0.9.1
	github.com/stretchr/testify v1.11.1
	github.com/x448/float16 v0.8.4
	golang.org/x/exp v0.0.0-20260218203240-3dfff04db8fa
	golang.org/x/text v0.34.0
	google.golang.org/protobuf v1.36.11
)

replace github.com/gomlx/gomlx => ../gomlx

require (
	github.com/davecgh/go-spew v1.1.2-0.20180830191138-d8f796af33cc // indirect
	github.com/go-logr/logr v1.4.3 // indirect
	github.com/pmezard/go-difflib v1.0.1-0.20181226105442-5d4384ee4fb2 // indirect
	golang.org/x/sys v0.41.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
	k8s.io/klog/v2 v2.130.1 // indirect
)

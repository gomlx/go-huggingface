package msmarco

const (
	ID              = "microsoft/ms_marco"
	Config          = "v2.1"
	TrainSplit      = "train"
	TestSplit       = "test"
	ValidationSplit = "validation"
)

//go:generate go tool generate_dataset_structs -dataset microsoft/ms_marco -config v2.1 -split validation -output gen_schema.go -fmt -package msmarco

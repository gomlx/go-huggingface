package dolly

const (
	ID         = "databricks/databricks-dolly-15k"
	Config     = "default"
	TrainSplit = "train"
)

//go:generate go tool generate_dataset_structs -dataset databricks/databricks-dolly-15k -config default -split train -root-struct Record -output gen_schema.go -fmt -package dolly

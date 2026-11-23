package dolly

type Record struct {
	Instruction string `json:"instruction" parquet:"instruction"`
	Context     string `json:"context" parquet:"context"`
	Response    string `json:"response" parquet:"response"`
	Category    string `json:"category" parquet:"category"`
}

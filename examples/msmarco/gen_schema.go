package msmarco

type MsMarcoRecord struct {
	Answers           []string     `json:"answers" parquet:"answers,list"`
	Passages          PassagesItem `json:"passages" parquet:"passages"`
	Query             string       `json:"query" parquet:"query"`
	QueryID           int32        `json:"query_id" parquet:"query_id"`
	QueryType         string       `json:"query_type" parquet:"query_type"`
	WellFormedAnswers []string     `json:"wellFormedAnswers" parquet:"wellFormedAnswers,list"`
}

type PassagesItem struct {
	IsSelected  []int32  `json:"is_selected" parquet:"is_selected,list"`
	PassageText []string `json:"passage_text" parquet:"passage_text,list"`
	URL         []string `json:"url" parquet:"url,list"`
}

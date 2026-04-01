package msmarco_test

import (
	"fmt"
	"testing"

	"github.com/gomlx/go-huggingface/datasets"
	"github.com/gomlx/go-huggingface/examples/msmarco"
)

func TestBrowse(t *testing.T) {
	ds := datasets.New(msmarco.ID)
	count := 0
	const limit = 10
	for record, err := range datasets.IterParquetFromDataset[msmarco.MsMarcoRecord](ds, msmarco.Config, msmarco.ValidationSplit) {
		if err != nil {
			t.Fatalf("Failed to get record: %+v", err)
		}
		fmt.Printf("Record #%02d: %s\n", count+1, record.Query)

		// List up to 3 passages.
		for i := range min(len(record.Passages.PassageText), 3) {
			passage := record.Passages.PassageText[i]
			fmt.Printf("- Passage #%d (of %d): %s\n", i, len(record.Passages.PassageText), passage)
		}

		fmt.Println()
		count++
		if count >= limit {
			break
		}

	}
}

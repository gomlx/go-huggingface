package hftokenizer

import "github.com/gomlx/go-huggingface/tokenizers/api"

// applyPostProcessor applies the post_processor to add special tokens
// (e.g., [CLS] prefix and [SEP] suffix for BERT-style models).
// This matches the Rust tokenizer's addSpecialTokens=true behavior.
//
// Supported types: TemplateProcessing, BertProcessing, RobertaProcessing.
func (t *Tokenizer) applyPostProcessor(ids []int, spans []api.TokenSpan) ([]int, []api.TokenSpan, []int) {
	pp := t.tokenizer.PostProcessor
	if pp == nil {
		return ids, spans, nil
	}

	switch pp.Type {
	case "TemplateProcessing":
		return t.applyTemplateProcessing(pp, ids, spans)
	case "BertProcessing", "RobertaProcessing":
		return t.applyBertProcessing(pp, ids, spans)
	default:
		return ids, spans, nil
	}
}

// applyTemplateProcessing handles TemplateProcessing post-processors.
func (t *Tokenizer) applyTemplateProcessing(pp *PostProcessor, ids []int, spans []api.TokenSpan) ([]int, []api.TokenSpan, []int) {
	if len(pp.Single) == 0 {
		return ids, spans, nil
	}

	var outIDs []int
	var outSpans []api.TokenSpan
	var outSpecial []int

	for _, item := range pp.Single {
		if item.SpecialToken != nil {
			st, ok := pp.SpecialTokens[item.SpecialToken.ID]
			if ok && len(st.IDs) > 0 {
				outIDs = append(outIDs, st.IDs...)
				for range st.IDs {
					outSpans = append(outSpans, api.TokenSpan{Start: -1, End: -1})
					outSpecial = append(outSpecial, 1)
				}
			}
		} else if item.Sequence != nil {
			outIDs = append(outIDs, ids...)
			outSpans = append(outSpans, spans...)
			for range ids {
				outSpecial = append(outSpecial, 0)
			}
		}
	}

	return outIDs, outSpans, outSpecial
}

// applyBertProcessing handles BertProcessing and RobertaProcessing post-processors.
// Format: {"type": "BertProcessing", "sep": ["[SEP]", 102], "cls": ["[CLS]", 101]}
func (t *Tokenizer) applyBertProcessing(pp *PostProcessor, ids []int, spans []api.TokenSpan) (outIDs []int, outSpans []api.TokenSpan, outSpecialMask []int) {
	clsID, hasCLS := parseTokenIDTuple(pp.Cls)
	sepID, hasSEP := parseTokenIDTuple(pp.Sep)

	if !hasCLS && !hasSEP {
		return ids, spans, nil
	}

	syntheticSpan := api.TokenSpan{Start: -1, End: -1}

	outIDs = make([]int, 0, len(ids)+2)
	outSpans = make([]api.TokenSpan, 0, len(ids)+2)
	outSpecialMask = make([]int, 0, len(ids)+2)

	if hasCLS {
		outIDs = append(outIDs, clsID)
		outSpans = append(outSpans, syntheticSpan)
		outSpecialMask = append(outSpecialMask, 1)
	}
	outIDs = append(outIDs, ids...)
	outSpans = append(outSpans, spans...)
	for range ids {
		outSpecialMask = append(outSpecialMask, 0)
	}
	if hasSEP {
		outIDs = append(outIDs, sepID)
		outSpans = append(outSpans, syntheticSpan)
		outSpecialMask = append(outSpecialMask, 1)
	}

	return outIDs, outSpans, outSpecialMask
}

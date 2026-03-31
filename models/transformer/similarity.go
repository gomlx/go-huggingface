package transformer

import (
	"strings"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/support/exceptions"
)

// Similiarity across all queries and documents, based on the model similarity configuration.
//
// - queries, documents: either [batchSize, embedDim] or for single query/documents [embedDim]
// - returns: [batchSizeQueries, batchSizeDocuments] matrix of similarities
func (m *Model) Similarity(queries, documents *Node) *Node {
	// Shape normalization: ensure they are at least 2-dimensional (batchSize, embedDim)
	if queries.Rank() == 1 {
		queries = ExpandDims(queries, 0)
	}
	if documents.Rank() == 1 {
		documents = ExpandDims(documents, 0)
	}
	if queries.Rank() != 2 || documents.Rank() != 2 {
		exceptions.Panicf("queries and documents must be 2-dimensional (batchSize, embedDim), got queries.Shape=%s, documents.Shape=%s",
			queries.Shape(), documents.Shape())
	}
	if queries.Shape().Dimensions[1] != documents.Shape().Dimensions[1] {
		exceptions.Panicf("queries and documents must have the same embedDim, got queries.Shape=%s, documents.Shape=%s",
			queries.Shape(), documents.Shape())
	}

	simFn := "cosine" // default
	if m.SentenceTransformerConfig != nil && m.SentenceTransformerConfig.SimilarityFnName != "" {
		simFn = m.SentenceTransformerConfig.SimilarityFnName
	}

	switch strings.ToLower(simFn) {
	case "dot", "dot_product":
		// Dot product (Matrix multiplication)
		return Einsum("qe,de->qd", queries, documents)

	case "euclidean":
		// Negative Euclidean distance
		qExpanded := ExpandDims(queries, 1)   // [batchQueries, 1, embedDim]
		dExpanded := ExpandDims(documents, 0) // [1, batchDocuments, embedDim]
		diff := Sub(qExpanded, dExpanded)
		dist := L2Norm(diff, -1)
		return Neg(dist)

	case "manhattan":
		// Negative Manhattan distance
		qExpanded := ExpandDims(queries, 1)
		dExpanded := ExpandDims(documents, 0)
		diff := Abs(Sub(qExpanded, dExpanded))
		dist := ReduceSum(diff, -1)
		return Neg(dist)

	case "cosine":
		fallthrough
	default:
		// Cosine similarity: dot product of L2-normalized vectors
		normQ := L2Normalize(queries, -1)
		normD := L2Normalize(documents, -1)
		return Einsum("qe,de->qd", normQ, normD)
	}
}

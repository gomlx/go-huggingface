// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package timestral

import (
	"math"
	"slices"
)

// DefaultTargetEncodingSmooth is the default smoothing factor for TargetEncode.
const DefaultTargetEncodingSmooth = 1.0

// TargetEncode performs per-series smoothed mean-target encoding of categorical covariates,
// matching timestral.covariate_encoding.target_encode.
//
// In a series with context mean M, category c is encoded as:
//
//	(smooth * M + sum_c) / (smooth + count_c)
//
// Missing values (NaN) form their own category. A category that never occurs in pastCodes
// (e.g., an unseen category in futureCodes) resolves to the series mean M.
func TargetEncode(targetContext, pastCodes, futureCodes []float64, smooth float64) (encodedPast, encodedFuture []float32) {
	tLen := len(targetContext)
	observed := make([]bool, tLen)
	var targetSum, itemCount float64
	for i, v := range targetContext {
		if !math.IsNaN(v) && !math.IsInf(v, 0) {
			observed[i] = true
			targetSum += v
			itemCount += 1.0
		}
	}
	var itemMean float64
	if itemCount > 0 {
		itemMean = targetSum / itemCount
	}

	// Collect unique sorted finite category codes from pastCodes.
	var finiteCodes []float64
	for _, c := range pastCodes {
		if !math.IsNaN(c) && !math.IsInf(c, 0) {
			finiteCodes = append(finiteCodes, c)
		}
	}
	slices.Sort(finiteCodes)
	categories := slices.Compact(finiteCodes)
	nanSlot := len(categories)

	sums := make([]float64, nanSlot+1)
	counts := make([]float64, nanSlot+1)
	pastSlots := make([]int, len(pastCodes))
	for i, c := range pastCodes {
		slot := nanSlot
		if !math.IsNaN(c) && !math.IsInf(c, 0) {
			idx, _ := slices.BinarySearch(categories, c)
			slot = idx
		}
		pastSlots[i] = slot
		if observed[i] {
			sums[slot] += targetContext[i]
			counts[slot] += 1.0
		}
	}

	unseenSlot := nanSlot + 1
	lookup := make([]float32, unseenSlot+1)
	for s := range nanSlot + 1 {
		lookup[s] = float32((smooth*itemMean + sums[s]) / (smooth + counts[s]))
	}
	lookup[unseenSlot] = float32(itemMean)

	encodedPast = make([]float32, len(pastCodes))
	for i, slot := range pastSlots {
		encodedPast[i] = lookup[slot]
	}

	if futureCodes != nil {
		encodedFuture = make([]float32, len(futureCodes))
		for i, c := range futureCodes {
			if math.IsNaN(c) {
				encodedFuture[i] = lookup[nanSlot]
			} else if !math.IsInf(c, 0) {
				if idx, found := slices.BinarySearch(categories, c); found {
					encodedFuture[i] = lookup[idx]
				} else {
					encodedFuture[i] = lookup[unseenSlot]
				}
			} else {
				encodedFuture[i] = lookup[unseenSlot]
			}
		}
	}
	return encodedPast, encodedFuture
}

// encodeCategoricalSeries encodes [B, C, L] float64 category codes against a univariate
// [B, 1, T] float32 context and bool mask over the last maxContext steps.
// When numVariates != 1, codes are simply converted to float32.
func encodeCategoricalSeries(
	codesFlat []float64,
	batchSize, numCatChannels, codeLen int,
	inputsFlat []float32,
	maskFlat []bool,
	numVariates, ctxLen, maxContext int,
) []float32 {
	out := make([]float32, len(codesFlat))
	if numVariates != 1 {
		for i, v := range codesFlat {
			out[i] = float32(v)
		}
		return out
	}

	winStart := max(0, ctxLen-maxContext)
	for b := range batchSize {
		baseCtx := b * ctxLen
		// Build forward-filled target context from first observation onwards within [winStart, ctxLen).
		targetCtx := make([]float64, ctxLen)
		started := false
		var lastVal float32
		for t := range ctxLen {
			obs := maskFlat[baseCtx+t] && !math.IsNaN(float64(inputsFlat[baseCtx+t])) && !math.IsInf(float64(inputsFlat[baseCtx+t]), 0)
			if obs {
				started = true
				lastVal = inputsFlat[baseCtx+t]
			}
			if started && t >= winStart {
				targetCtx[t] = float64(lastVal)
			} else {
				targetCtx[t] = math.NaN()
			}
		}

		for ch := range numCatChannels {
			baseCode := (b*numCatChannels + ch) * codeLen
			pastCodes := codesFlat[baseCode : baseCode+ctxLen]
			var futureCodes []float64
			if codeLen > ctxLen {
				futureCodes = codesFlat[baseCode+ctxLen : baseCode+codeLen]
			}
			pastEnc, futureEnc := TargetEncode(targetCtx, pastCodes, futureCodes, DefaultTargetEncodingSmooth)
			for t := range ctxLen {
				if math.IsNaN(targetCtx[t]) && math.IsNaN(pastCodes[t]) {
					out[baseCode+t] = float32(math.NaN())
				} else {
					out[baseCode+t] = pastEnc[t]
				}
			}
			if futureEnc != nil {
				copy(out[baseCode+ctxLen:baseCode+codeLen], futureEnc)
			}
		}
	}
	return out
}

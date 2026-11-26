// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build !linux

package modeltests

func resetKernelPeakRSS() bool {
	return false
}

func currentRSSBytes() int64 {
	return 0
}

func kernelPeakRSSBytes() int64 {
	return 0
}

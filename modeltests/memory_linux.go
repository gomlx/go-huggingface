// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build linux

package modeltests

import (
	"bufio"
	"os"
	"strconv"
	"strings"

	"k8s.io/klog/v2"
)

// resetKernelPeakRSS resets /proc/self/status VmHWM on Linux (kernel >= 4.0)
// by writing "5" to /proc/self/clear_refs. Returns true if the reset succeeded.
func resetKernelPeakRSS() bool {
	return os.WriteFile("/proc/self/clear_refs", []byte("5\n"), 0) == nil
}

func currentRSSBytes() int64 {
	return readProcStatusKB("VmRSS:") * 1024
}

func kernelPeakRSSBytes() int64 {
	return readProcStatusKB("VmHWM:") * 1024
}

func readProcStatusKB(prefix string) int64 {
	const statusFile = "/proc/self/status"
	f, err := os.Open(statusFile)
	if err != nil {
		return 0
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, prefix) {
			fields := strings.Fields(line[len(prefix):])
			if len(fields) >= 1 {
				if v, err := strconv.ParseInt(fields[0], 10, 64); err == nil {
					return v
				}
			}
			return 0
		}
	}
	if err := scanner.Err(); err != nil {
		klog.Errorf("Failed to read %s: %+v", statusFile, err)
		return 0
	}
	return 0
}

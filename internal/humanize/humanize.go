package humanize

import "fmt"

// Bytes returns the rendering of bytes aproximated to the nearest power of 1024 (Kb, Mb, Gb, Tb, etc.)
// with one decimal place.
func Bytes(num int64) string {
	sign := ""
	if num < 0 {
		sign = "-"
		num = -num
	}
	if num < 1024 {
		return fmt.Sprintf("%s%d B", sign, num)
	}
	const unit = 1024
	div, exp := int64(unit), 0
	for n := num / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%s%.1f %cb", sign, float64(num)/float64(div), "KMGTPE"[exp])
}

// Count returns a compact string representation of an integer count appending powers of 1024 suffixes (K, M, G, T, P, E).
func Count(num int64) string {
	sign := ""
	if num < 0 {
		sign = "-"
		num = -num
	}
	if num < 1024 {
		return fmt.Sprintf("%s%d", sign, num)
	}
	const unit = 1024
	div, exp := int64(unit), 0
	for n := num / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	res := fmt.Sprintf("%s%.1f%c", sign, float64(num)/float64(div), "KMGTPE"[exp])
	if len(res) > 3 && res[len(res)-3:len(res)-1] == ".0" {
		res = res[:len(res)-3] + res[len(res)-1:]
	}
	return res
}

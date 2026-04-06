package humanize

import (
	"fmt"
	"math"
	"regexp"
	"strconv"
	"time"
)

// EraseToEndOfLine is the ANSI control sequence to erase the current line from the cursor to the end.
const EraseToEndOfLine = "\033[K"

// Bytes returns the rendering of bytes aproximated to the nearest power of 1024 (Kb, Mb, Gb, Tb, etc.)
// with one decimal place.
func Bytes[T ~int64 | ~int32 | ~int | ~uint64 | ~uint32 | ~uint](numT T) string {
	num := int64(numT)
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

// Count returns a compact string representation of an integer count appending powers of 1000 suffixes (K, M, G, T, P, E).
func Count[T ~int64 | ~int32 | ~int | ~uint64 | ~uint32 | ~uint](numT T) string {
	num := int64(numT)
	sign := ""
	if num < 0 {
		sign = "-"
		num = -num
	}
	const unit = 1000
	if num < unit {
		return fmt.Sprintf("%s%d", sign, num)
	}
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

// Speed returns some human readable speed (or ratio) of some unit / second.
func Speed[T ~float64 | ~float32](ratioT T, unit string) string {
	ratio := float64(ratioT)
	if ratio > 10 {
		return fmt.Sprintf("%s %s/s", Count(int64(math.Round(ratio))), unit)
	}
	return fmt.Sprintf("%.2f %s/s", ratio, unit)
}

// Duration pretty prints duration without a long list of decimal points.
func Duration(d time.Duration) string {
	s := d.String()
	re := regexp.MustCompile(`(\d+\.?\d*)([µa-z]+)`)
	matches := re.FindStringSubmatch(s)
	if len(matches) != 3 {
		return s
	}
	num, err := strconv.ParseFloat(matches[1], 64)
	if err != nil {
		return s
	}
	return fmt.Sprintf("%.2f%s", num, matches[2])
}

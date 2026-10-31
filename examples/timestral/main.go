// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Command timestral times native GoMLX Timestral forecasts loaded directly from Hugging Face
// safetensors via models/timestral, optionally comparing accuracy against a reference .f32 file.
//
// It compiles one (batch=2, contextSteps) input shape once with weights baked as graph constants
// (batch 2 = [context, -context] for symmetric averaging).
//
//	GOEXPERIMENT=simd go run ./examples/timestral -model mistralai/Timestral-28M -compare ./examples/timestral/model-28m_test_results.f32 -runs 5
package main

import (
	_ "embed"
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"syscall"
	"time"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	_ "github.com/gomlx/compute/gobackend" // registers the pure Go backend "go"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/models/timestral"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	mlmodel "github.com/gomlx/gomlx/ml/model"
)

//go:embed context.f32
var embeddedContextBytes []byte

func main() {
	obs := flag.String("obs", "", "Optional observation label for the rows printed")
	modelID := flag.String("model", timestral.RepoTimestral28M, "Hugging Face model ID or local checkpoint directory")
	compare := flag.String("compare", "", "Optional path to reference .f32 results file to compare against")
	contextSteps := flag.Int("context-steps", 12096, "Number of context steps (default: 42 days at 5-min intervals)")
	horizonSteps := flag.Int("horizon-steps", 864, "Number of horizon steps (default: 72 hours at 5-min intervals)")
	backendConfig := flag.String("backend", "go", "GoMLX backend config")
	runs := flag.Int("runs", 5, "Timed runs after the first (warm-up) one")
	flag.Parse()

	context := loadOrGenerateContext(*contextSteps)
	var reference []float32
	if *compare != "" {
		reference = readF32(*compare)
	}

	const batch = 2
	input := make([]float32, batch*(*contextSteps))
	copy(input, context)
	for i, v := range context {
		input[*contextSteps+i] = -v
	}

	t0 := time.Now()
	backend := must1(compute.NewWithConfig(*backendConfig))
	repo := openRepo(*modelID)
	tmModel := must1(timestral.LoadModel(repo))
	numQuantiles := tmModel.Config.NumQuantiles()

	fmt.Printf("%s: context %d, horizon %d, %d quantiles, batch %d, backend %q, GOMAXPROCS %d\n",
		*modelID, *contextSteps, *horizonSteps, numQuantiles, batch, *backendConfig, runtime.GOMAXPROCS(0))

	// Load weights into store with WithVariableAsConst(true) so weights are baked into the compiled graph.
	store := mlmodel.NewStore().WithVariableAsConst(true)
	must(tmModel.LoadStore(backend, store))

	exec := must1(mlmodel.NewExec(backend, store, func(scope *mlmodel.Scope, x *graph.Node) *graph.Node {
		return tmModel.BuildGraph(scope, x, *horizonSteps).
			SymmetricAveraging(false).
			ClipNonNegative(false).
			Done()
	}))

	// Compile only the (batch, contextSteps) shape, then release the variable store.
	must1(exec.Compile(shapes.Make(dtypes.Float32, batch, *contextSteps)))
	store.Finalize()
	debug.FreeOSMemory()
	fmt.Printf("load + compile: %.2fs, max RSS %.2f GB\n", time.Since(t0).Seconds(), maxRSS())

	labelName := filepath.Base(*modelID)

	fmt.Println()
	fmt.Println("| Obs | Model | Run | Wall | CPU Time | Cores | Max Abs Diff vs PyTorch | Max RSS |")
	fmt.Println("| --- | --- | :--- | ---: | ---: | ---: | ---: | ---: |")
	var (
		bestDt    time.Duration
		bestDc    time.Duration
		bestCheck string
		bestRSS   float64
	)
	for r := range *runs + 1 {
		tStart, cStart := time.Now(), cpuTime()
		x := tensors.FromFlatDataAndDimensions(input, batch, *contextSteps)
		y := must1(exec.Call1(x))
		got := must1(tensors.CopyFlatData[float32](y))
		dt, dc := time.Since(tStart), cpuTime()-cStart
		x.FinalizeAll()
		y.FinalizeAll()

		check := "-"
		if len(reference) == len(got) {
			diff := 0.0
			for i, v := range got {
				diff = max(diff, math.Abs(float64(v)-float64(reference[i])))
			}
			check = fmt.Sprintf("%.2g", diff)
		}
		rss := maxRSS()
		if r >= 1 && (bestDt == 0 || dt < bestDt) {
			bestDt = dt
			bestDc = dc
			bestCheck = check
			bestRSS = rss
		}
		label := fmt.Sprintf("run %d", r)
		if r == 0 {
			label = "warm-up"
		}
		fmt.Printf("| %s | %s | %-7s | %.3fs | %.3fs | %.1f | %s | %.2f GB |\n",
			*obs, labelName, label, dt.Seconds(), dc.Seconds(), dc.Seconds()/dt.Seconds(), check, rss)
	}
	if *runs >= 1 {
		fmt.Printf("| %s | %s | %-7s | %.3fs | %.3fs | %.1f | %s | %.2f GB |\n",
			*obs, labelName, "best", bestDt.Seconds(), bestDc.Seconds(), bestDc.Seconds()/bestDt.Seconds(), bestCheck, bestRSS)
	}
}

func openRepo(modelOrDir string) *hub.Repo {
	if info, err := os.Stat(modelOrDir); err == nil && info.IsDir() {
		return hub.NewLocal(modelOrDir)
	}
	repo := hub.New(modelOrDir)
	must(repo.FetchFiles("config.json", "model.safetensors"))
	return repo
}

func loadOrGenerateContext(steps int) []float32 {
	if len(embeddedContextBytes) == steps*4 {
		return decodeF32Bytes(embeddedContextBytes)
	}
	return syntheticContext(steps)
}

func syntheticContext(steps int) []float32 {
	const stepsPerDay = 288
	rng := rand.New(rand.NewPCG(0, 1))
	out := make([]float32, steps)
	for t := range steps {
		tf := float64(t)
		val := 20.0 + 5.0*math.Sin(2*math.Pi*tf/stepsPerDay) + 0.5*math.Sin(2*math.Pi*tf/(7*stepsPerDay))
		val += rng.NormFloat64() * 0.2
		out[t] = float32(val)
	}
	for i := max(0, steps-500); i < max(0, steps-300); i++ {
		out[i] = float32(math.NaN())
	}
	return out
}

func readF32(path string) []float32 {
	return decodeF32Bytes(must1(os.ReadFile(path)))
}

func decodeF32Bytes(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

func maxRSS() float64 {
	var ru syscall.Rusage
	_ = syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	if runtime.GOOS == "darwin" {
		return float64(ru.Maxrss) / 1e9 // bytes
	}
	return float64(ru.Maxrss) * 1024 / 1e9 // KiB
}

func cpuTime() time.Duration {
	var ru syscall.Rusage
	_ = syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	return time.Duration(ru.Utime.Nano() + ru.Stime.Nano())
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}

func must1[T any](v T, err error) T {
	must(err)
	return v
}

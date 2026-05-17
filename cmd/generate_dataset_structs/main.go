package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"sort"
	"strings"

	"github.com/gomlx/go-huggingface/datasets"
	"k8s.io/klog/v2"
)

var (
	datasetFlag = flag.String("dataset", "", "dataset name to extract the info from, e.g.: \"microsoft/ms_marco\"")
	configFlag  = flag.String("config", "", "config of the dataset to use. This is sometimes used for versioning, e.g.: \"v2.1\". If empty, it will be automatically chosen if only one exists. If multiple configs exist, lists them and exits.")
	splitFlag   = flag.String("split", "", "split of the dataset to use, e.g.: \"train\". If empty, it will be automatically chosen if only one exists. If multiple splits exist, lists them and exits.")
	outputFlag  = flag.String("output", "", "file name where to output the generated Go code. If empty, the structs are output to stdout. It requires -package to be set also.")
	packageFlag = flag.String("package", "", "name of the package to use when outputting a .go file with output")
	fmtFlag     = flag.Bool("fmt", true, "run go fmt on the output file after generating it")
)

func main() {
	flag.Parse()

	if *datasetFlag == "" {
		klog.Fatal("Missing required flag: -dataset")
	}

	if *outputFlag != "" && *packageFlag == "" {
		klog.Fatal("If -output is specified, -package is also required")
	}

	klog.V(1).Infof("Loading dataset %q info...", *datasetFlag)
	ds := datasets.New(*datasetFlag)
	info, err := ds.Info()
	if err != nil {
		klog.Fatalf("Failed to retrieve dataset info: %+v", err)
	}
	if len(info.DatasetInfo) == 0 {
		klog.Fatal("Failed to retrieve dataset info or dataset has no configurations")
	}

	config := *configFlag
	if config == "" {
		if len(info.DatasetInfo) == 1 {
			// Choose the only config
			for k := range info.DatasetInfo {
				config = k
			}
			klog.Infof("Automatically selected the only available config: %q", config)
		} else {
			// List available configs
			var configs []string
			for k := range info.DatasetInfo {
				configs = append(configs, k)
			}
			sort.Strings(configs)
			fmt.Printf("Multiple configs available: \"%s\"\n", strings.Join(configs, "\", \""))
			err = ds.DownloadParquetFilesInfo(model.Background(), false)
			if err != nil {
				klog.Fatalf("Failed to download parquet files info: %+v", err)
			}
			fmt.Printf("%s\n", ds)
			os.Exit(1)
		}
	}

	configInfo, ok := info.DatasetInfo[config]
	if !ok {
		klog.Fatalf("Config %q not found in dataset info.", config)
	}

	split := *splitFlag
	if split == "" {
		if len(configInfo.Splits) == 1 {
			for k := range configInfo.Splits {
				split = k
			}
			klog.Infof("Automatically selected the only available split: %q", split)
		} else {
			var splits []string
			for k := range configInfo.Splits {
				splits = append(splits, k)
			}
			sort.Strings(splits)
			fmt.Printf("Multiple splits available for config %q: \"%s\"\n", config, strings.Join(splits, "\", \""))
			os.Exit(1)
		}
	}

	generatedCode, err := ds.GenerateGoStruct(config, split)
	if err != nil {
		klog.Fatalf("Failed to generate Go struct: %+v", err)
	}

	var output strings.Builder
	if *packageFlag != "" {
		output.WriteString(fmt.Sprintf("package %s\n\n", *packageFlag))
	} else if *outputFlag != "" {
		// Fallback just in case, though the check above fails if -package is missing when -output is present
		output.WriteString("package main\n\n")
	}
	output.WriteString(generatedCode)

	if *outputFlag == "" {
		fmt.Print(output.String())
	} else {
		err := os.WriteFile(*outputFlag, []byte(output.String()), 0644)
		if err != nil {
			klog.Fatalf("Failed to write output file: %v", err)
		}
		if *fmtFlag {
			cmd := exec.Command("go", "fmt", *outputFlag)
			if err := cmd.Run(); err != nil {
				klog.Warningf("Failed to run go fmt on output file: %v", err)
			}
		}
		fmt.Printf("✅ generate_dataset_structs:       \tsuccessfully generated %s\n", *outputFlag)

	}
}

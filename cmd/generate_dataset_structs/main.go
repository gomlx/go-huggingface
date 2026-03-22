package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"sort"
	"strings"

	"github.com/gomlx/go-huggingface/datasets"
)

var (
	datasetFlag = flag.String("dataset", "", "dataset name to extract the info from, e.g.: \"microsoft/ms_marco\"")
	versionFlag = flag.String("version", "", "version of the dataset to use, e.g.: \"v2.1\". If empty, it will be automatically chosen if only one exists. If multiple versions exist, lists them and exits.")
	outputFlag  = flag.String("output", "", "file name where to output the generated Go code. If empty, the structs are output to stdout. It requires -package to be set also.")
	packageFlag = flag.String("package", "", "name of the package to use when outputting a .go file with output")
)

func main() {
	flag.Parse()

	if *datasetFlag == "" {
		log.Fatal("Missing required flag: -dataset")
	}

	if *outputFlag != "" && *packageFlag == "" {
		log.Fatal("If -output is specified, -package is also required")
	}

	log.Printf("Loading dataset %q info...", *datasetFlag)
	ds := datasets.New(*datasetFlag)
	info := ds.Info()
	if info == nil || len(info.DatasetInfo) == 0 {
		log.Fatal("Failed to retrieve dataset info or dataset has no configurations")
	}

	version := *versionFlag
	if version == "" {
		if len(info.DatasetInfo) == 1 {
			// Choose the only version
			for k := range info.DatasetInfo {
				version = k
			}
			log.Printf("Automatically selected the only available version: %q", version)
		} else {
			// List available versions
			var versions []string
			for k := range info.DatasetInfo {
				versions = append(versions, k)
			}
			sort.Strings(versions)
			log.Fatalf("Multiple versions available: %s. Please specify one using -version flag.", strings.Join(versions, ", "))
		}
	}

	configInfo, ok := info.DatasetInfo[version]
	if !ok {
		log.Fatalf("Version %q not found in dataset info.", version)
	}

	// Generate a root struct name from the dataset name
	parts := strings.Split(*datasetFlag, "/")
	namePart := parts[len(parts)-1]
	rootStructName := toCamelCase(namePart) + "Record"

	generatedCode := configInfo.GenerateGoStruct(rootStructName)

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
			log.Fatalf("Failed to write output file: %v", err)
		}
		log.Printf("Successfully generated Go structs to %s", *outputFlag)
	}
}

func toCamelCase(s string) string {
	parts := strings.Split(s, "_")
	for i := range parts {
		if len(parts[i]) > 0 {
			parts[i] = strings.ToUpper(parts[i][:1]) + parts[i][1:]
		}
	}
	return strings.Join(parts, "")
}

package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/gomlx/go-huggingface/datasets"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

var (
	datasetFlag    = flag.String("dataset", "", "dataset name to extract the info from, e.g.: \"microsoft/ms_marco\"")
	configFlag     = flag.String("config", "", "config of the dataset to use. This is sometimes used for versioning, e.g.: \"v2.1\". If empty, it will be automatically chosen if only one exists. If multiple configs exist, lists them and exits.")
	splitFlag      = flag.String("split", "", "split of the dataset to use, e.g.: \"train\". If empty, it will be automatically chosen if only one exists. If multiple splits exist, lists them and exits.")
	parquetFlag    = flag.String("parquet", "", "local path to a parquet file to generate structs directly from, bypassing the HuggingFace datasets server API")
	rootStructFlag = flag.String("root-struct", "", "name of the root struct to generate (optional, overrides default name)")
	outputFlag     = flag.String("output", "", "file name where to output the generated Go code. If empty, the structs are output to stdout. It requires -package to be set also.")
	packageFlag    = flag.String("package", "", "name of the package to use when outputting a .go file with output")
	fmtFlag        = flag.Bool("fmt", true, "run go fmt on the output file after generating it")
	colorFlag      = flag.Bool("color", true, "enable colorized terminal output using lipgloss")
)

func main() {
	flag.Parse()

	validateFlags()

	var generatedCode string
	var err error
	var rootStruct string

	if *parquetFlag != "" {
		rootStruct = getRootStructName(*parquetFlag)
		generatedCode, err = generateFromLocalParquet(*parquetFlag, rootStruct)
		if err != nil {
			klog.Fatalf("Failed to generate Go struct from parquet: %+v", err)
		}
	} else {
		rootStruct = getRootStructName(*datasetFlag)

		// 1. Attempt to find a cached parquet file matching config & split first
		cachedPath := findCachedParquetFile(*datasetFlag, *configFlag, *splitFlag)
		if cachedPath != "" {
			klog.Infof("Found cached parquet file: %q", cachedPath)
			generatedCode, err = generateFromLocalParquet(cachedPath, rootStruct)
			if err == nil {
				handleOutput(generatedCode)
				return
			}
			klog.Warningf("Failed to generate from cached file %q: %+v. Proceeding with standard flow...", cachedPath, err)
		}

		// 2. Fall back to standard datasets-server flow
		generatedCode, err = generateFromDatasetServer(*datasetFlag, *configFlag, *splitFlag)
		if err != nil {
			klog.V(2).Infof("Failed to retrieve dataset info or generate struct via datasets-server API: %+v", err)
			klog.Infof("Attempting to automatically locate parquet files in the repository...")

			var localPath string
			localPath, rootStruct, err = fallbackLocateAndDownloadParquet(*datasetFlag)
			if err != nil {
				klog.Fatalf("Fallback failed: %+v", err)
			}

			generatedCode, err = generateFromLocalParquet(localPath, rootStruct)
			if err != nil {
				klog.Fatalf("Failed to generate Go struct: %+v", err)
			}
		}
	}

	handleOutput(generatedCode)
}

func validateFlags() {
	if *datasetFlag == "" && *parquetFlag == "" {
		klog.Fatal("Missing required flag: -dataset or -parquet must be specified")
	}

	if *outputFlag != "" && *packageFlag == "" {
		klog.Fatal("If -output is specified, -package is also required")
	}
}

func getRootStructName(source string) string {
	if *rootStructFlag != "" {
		return *rootStructFlag
	}
	var namePart string
	if *datasetFlag != "" {
		parts := strings.Split(*datasetFlag, "/")
		namePart = parts[len(parts)-1]
	} else {
		base := filepath.Base(source)
		base = strings.TrimSuffix(base, filepath.Ext(base))
		namePart = base
	}
	return datasets.ToCamelCase(namePart) + "Record"
}

func generateFromLocalParquet(path string, rootStruct string) (string, error) {
	klog.Infof("Generating Go struct from local parquet file %q with root struct %q...", path, rootStruct)
	return datasets.GenerateGoStructFromParquet(path, rootStruct)
}

func generateFromDatasetServer(dataset string, config string, split string) (string, error) {
	klog.Infof("Loading dataset %q info...", dataset)
	ds := datasets.New(dataset)
	info, err := ds.Info()
	if err != nil {
		return "", err
	}
	if len(info.DatasetInfo) == 0 {
		return "", errors.New("dataset has no configurations")
	}

	if config == "" {
		var configs []string
		for k := range info.DatasetInfo {
			configs = append(configs, k)
		}
		sort.Strings(configs)

		if len(configs) > 1 {
			klog.Infof("Available configurations: %s", strings.Join(configs, ", "))
		}
		config = configs[0]
		klog.Infof("Automatically selected configuration: %q", config)
	}

	configInfo, ok := info.DatasetInfo[config]
	if !ok {
		return "", errors.Errorf("config %q not found in dataset info", config)
	}

	if split == "" {
		var splits []string
		for k := range configInfo.Splits {
			splits = append(splits, k)
		}
		sort.Strings(splits)

		if len(splits) > 1 {
			klog.Infof("Available splits for config %q: %s", config, strings.Join(splits, ", "))
		}
		split = splits[0]
		klog.Infof("Automatically selected split: %q", split)
	}

	return ds.GenerateGoStruct(config, split)
}

func findCachedParquetFile(datasetID string, config string, split string) string {
	ds := datasets.New(datasetID)
	cacheDir, err := ds.CacheDir()
	if err != nil {
		return ""
	}

	var foundPath string
	_ = filepath.Walk(cacheDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if !info.IsDir() && strings.HasSuffix(info.Name(), ".parquet") {
			// If config is specified, it must be in the path segment (e.g. /v2.1/ or /v2.1.parquet)
			if config != "" && !strings.Contains(path, "/"+config+"/") && !strings.Contains(path, "_"+config+"_") {
				return nil
			}
			// If split is specified, it must be in the path segment
			if split != "" && !strings.Contains(path, "/"+split+"/") && !strings.Contains(path, "_"+split+"_") {
				return nil
			}
			foundPath = path
			return filepath.SkipAll
		}
		return nil
	})
	return foundPath
}

func fallbackLocateAndDownloadParquet(dataset string) (localPath string, rootStruct string, err error) {
	ds := datasets.New(dataset)
	var parquetFiles []string
	for file, loopErr := range ds.IterFileNames() {
		if loopErr != nil {
			return "", "", errors.Wrap(loopErr, "failed to list files from repository")
		}
		if strings.HasSuffix(file, ".parquet") {
			parquetFiles = append(parquetFiles, file)
		}
	}
	if len(parquetFiles) == 0 {
		return "", "", errors.New("no parquet files found in dataset repository")
	}

	fmt.Printf("[INFO] Found %d parquet files in repository.\n", len(parquetFiles))
	
	if len(parquetFiles) > 10 {
		fmt.Printf("[INFO] There are %d parquet files in the repository. We automatically picked file %s\n",
			len(parquetFiles),
			parquetFiles[0],
		)
	} else if len(parquetFiles) > 1 {
		fmt.Printf("[INFO] Available parquet files in repository:\n")
		for i, pf := range parquetFiles {
			fmt.Printf("  [%d] %s\n", i, pf)
		}
		fmt.Printf("[INFO] Automatically selected first parquet file: %s\n", parquetFiles[0])
	} else {
		fmt.Printf("[INFO] Automatically selected the only parquet file: %s\n", parquetFiles[0])
	}

	selectedFile := parquetFiles[0]
	rootStruct = getRootStructName(selectedFile)
	fmt.Printf("[INFO] Automatically selected root-struct: %s\n", rootStruct)

	klog.Infof("Downloading parquet file %q...", selectedFile)
	localPath, err = ds.DownloadFile(selectedFile)
	if err != nil {
		return "", "", errors.Wrap(err, "failed to download parquet file")
	}
	return localPath, rootStruct, nil
}

func handleOutput(generatedCode string) {
	var output strings.Builder
	if *packageFlag != "" {
		output.WriteString(fmt.Sprintf("package %s\n\n", *packageFlag))
	} else if *outputFlag != "" {
		output.WriteString("package main\n\n")
	}
	output.WriteString(generatedCode)

	if *outputFlag == "" {
		fmt.Print(highlightGoCode(output.String()))
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
		klog.Infof("generate_dataset_structs: successfully generated %s", *outputFlag)
	}
}

func highlightGoCode(code string) string {
	if !*colorFlag {
		return code
	}
	
	styleKeyword  := lipgloss.NewStyle().Foreground(lipgloss.Color("204")).Bold(true) // Magenta for keywords
	styleStruct   := lipgloss.NewStyle().Foreground(lipgloss.Color("39")).Bold(true)  // Blue for structs
	styleField    := lipgloss.NewStyle().Foreground(lipgloss.Color("252"))            // White for fields
	styleType     := lipgloss.NewStyle().Foreground(lipgloss.Color("114"))            // Green for types
	styleTagKey   := lipgloss.NewStyle().Foreground(lipgloss.Color("215"))            // Orange/Yellow for tag names
	styleTagValue := lipgloss.NewStyle().Foreground(lipgloss.Color("117"))            // Cyan for tag values
	stylePunct    := lipgloss.NewStyle().Foreground(lipgloss.Color("243"))            // Muted gray for symbols
	styleComment  := lipgloss.NewStyle().Foreground(lipgloss.Color("244")).Italic(true) // Gray italic for comments
	
	lines := strings.Split(code, "\n")
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		
		if strings.HasPrefix(trimmed, "//") {
			lines[i] = styleComment.Render(line)
			continue
		}
		
		if strings.HasPrefix(trimmed, "package ") {
			packageName := strings.TrimPrefix(trimmed, "package ")
			packageName = strings.TrimSpace(packageName)
			lines[i] = styleKeyword.Render("package") + " " + packageName
			continue
		}
		
		if strings.HasPrefix(trimmed, "type ") && strings.HasSuffix(trimmed, " struct {") {
			structName := strings.TrimPrefix(trimmed, "type ")
			structName = strings.TrimSuffix(structName, " struct {")
			structName = strings.TrimSpace(structName)
			
			indent := len(line) - len(strings.TrimLeft(line, " \t"))
			indentStr := line[:indent]
			
			lines[i] = indentStr + styleKeyword.Render("type") + " " +
				styleStruct.Render(structName) + " " +
				styleKeyword.Render("struct") + " " +
				stylePunct.Render("{")
			continue
		}
		
		if trimmed == "}" {
			indent := len(line) - len(strings.TrimLeft(line, " \t"))
			indentStr := line[:indent]
			lines[i] = indentStr + stylePunct.Render("}")
			continue
		}
		
		if strings.Contains(line, "`") {
			parts := strings.SplitN(line, "`", 2)
			fieldAndType := parts[0]
			tags := parts[1]
			
			indent := len(fieldAndType) - len(strings.TrimLeft(fieldAndType, " \t"))
			indentStr := fieldAndType[:indent]
			fieldAndTypeTrim := strings.TrimSpace(fieldAndType)
			
			ftParts := strings.SplitN(fieldAndTypeTrim, " ", 2)
			if len(ftParts) == 2 {
				fieldName := ftParts[0]
				fieldType := ftParts[1]
				
				tags = strings.TrimSuffix(tags, "`")
				tagPairs := strings.Split(tags, " ")
				var styledTagPairs []string
				for _, pair := range tagPairs {
					if strings.Contains(pair, ":") {
						kv := strings.SplitN(pair, ":", 2)
						k := kv[0]
						v := kv[1]
						styledTagPairs = append(styledTagPairs,
							styleTagKey.Render(k) + stylePunct.Render(":") + styleTagValue.Render(v))
					} else {
						styledTagPairs = append(styledTagPairs, pair)
					}
				}
				
				lines[i] = indentStr +
					styleField.Render(fieldName) + " " +
					styleType.Render(fieldType) + " " +
					stylePunct.Render("`") + strings.Join(styledTagPairs, " ") + stylePunct.Render("`")
				continue
			}
		}
	}
	return strings.Join(lines, "\n")
}

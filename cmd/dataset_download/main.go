package main

import (
	"cmp"
	"context"
	"flag"
	"fmt"
	"os"
	"slices"

	"github.com/gomlx/compute/support/humanize"
	"github.com/gomlx/go-huggingface/datasets"
	"k8s.io/klog/v2"
)

var (
	datasetFlag = flag.String("dataset", "", "HuggingFace dataset ID, e.g. \"HuggingFaceFW/fineweb\"")
	configFlag  = flag.String("config", "", "Dataset configuration. If empty, lists available configs/splits. "+
		"Set to \"*\" to include all configs.")
	splitFlag = flag.String("split", "", "Dataset split (e.g. \"train\", \"validation\"). "+
		"If empty, lists available configs/splits. Set to \"*\" to include all splits.")
	listFlag      = flag.Bool("list", false, "List downloaded local cache files matching -config and -split instead of downloading")
	deleteFlag    = flag.Bool("delete", false, "Delete local cache files matching -config and -split instead of downloading")
	verbosityFlag = flag.Int("verbosity", 1, "Verbosity level for download progress")
)

func main() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] [dataset_id]\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Example: %s -config default -split train HuggingFaceFW/fineweb\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Flags:\n")
		flag.PrintDefaults()
	}
	flag.Parse()

	// Parse dataset ID from flag or positional argument
	datasetID := *datasetFlag
	if datasetID == "" && flag.NArg() > 0 {
		datasetID = flag.Arg(0)
	}

	if datasetID == "" {
		fmt.Fprintf(os.Stderr, "Error: missing dataset ID.\n\n")
		flag.Usage()
		os.Exit(1)
	}

	// Initialize dataset
	hfAuthToken := os.Getenv("HF_TOKEN")
	ds := datasets.New(datasetID).WithAuth(hfAuthToken)
	ds.Verbosity = *verbosityFlag

	config := *configFlag
	if config == "*" {
		config = ""
	}
	split := *splitFlag
	if split == "*" {
		split = ""
	}

	if *listFlag {
		listDownloadedFiles(ds, config, split)
		return
	}

	// Delete requires both config and split to be explicitly set (even if to "*")
	if *deleteFlag && (*configFlag == "" || *splitFlag == "") {
		fmt.Fprintf(os.Stderr, "Error: when using -delete, you must explicitly specify both -config and -split (use \"*\" to match all).\n")
		os.Exit(1)
	}

	if *deleteFlag {
		deleteLocalCache(ds, config, split)
		return
	}

	// If the user doesn't set a -config or a -split, list the available configs/splits
	if *configFlag == "" || *splitFlag == "" {
		listConfigsAndSplits(ds)
		return
	}

	// Both config and split are provided. Download all files
	ctx := context.Background()
	fmt.Printf("Downloading files for dataset %q, config %q, split %q...\n", datasetID, *configFlag, *splitFlag)
	downloadedPaths, err := ds.DownloadAll(ctx, config, split)
	if err != nil {
		klog.Fatalf("Error downloading files: %+v", err)
	}

	fmt.Printf("\nSuccessfully downloaded %d files:\n", len(downloadedPaths))
	for _, p := range downloadedPaths {
		fmt.Printf("  - %s\n", p)
	}
}

type ConfigSplit struct {
	Config string
	Split  string
}

type GroupInfo struct {
	Config string
	Split  string
	Count  int
	Size   int64
}

func listConfigsAndSplits(ds *datasets.Dataset) {
	fmt.Printf("Retrieving information for dataset %q...\n", ds.ID)

	parquetFiles, err := ds.GetParquetFiles()
	if err != nil {
		klog.Fatalf("Error retrieving parquet files: %+v", err)
	}

	if len(parquetFiles) == 0 {
		fmt.Printf("No files found for dataset %q.\n", ds.ID)
		return
	}

	// Group files by config and split
	groupsMap := make(map[ConfigSplit]*GroupInfo)
	for _, f := range parquetFiles {
		key := ConfigSplit{Config: f.Config, Split: f.Split}
		gi, exists := groupsMap[key]
		if !exists {
			gi = &GroupInfo{
				Config: f.Config,
				Split:  f.Split,
			}
			groupsMap[key] = gi
		}
		gi.Count++
		gi.Size += f.Size
	}

	// Sort groups by Config, then by Split
	var groups []*GroupInfo
	for _, gi := range groupsMap {
		groups = append(groups, gi)
	}
	slices.SortFunc(groups, func(a, b *GroupInfo) int {
		if a.Config != b.Config {
			return cmp.Compare(a.Config, b.Config)
		}
		return cmp.Compare(a.Split, b.Split)
	})

	fmt.Printf("\nAvailable configurations/splits for %q:\n", ds.ID)
	for _, gi := range groups {
		fmt.Printf("  -config %-25s -split %-15s (%d files, total size: %s)\n",
			gi.Config,
			gi.Split,
			gi.Count,
			humanize.Bytes(uint64(gi.Size)),
		)
	}
}

func deleteLocalCache(ds *datasets.Dataset, config, split string) {
	fmt.Printf("Locating local files for dataset %q, config %q, split %q to delete...\n",
		ds.ID,
		cmp.Or(config, "*"),
		cmp.Or(split, "*"),
	)

	localFiles, err := ds.ListDownloadedFiles(config, split)
	if err != nil {
		klog.Fatalf("Error listing local files: %+v", err)
	}

	if len(localFiles) == 0 {
		fmt.Println("No local cache files found matching the criteria.")
		return
	}

	deletedCount := 0
	var deletedBytes int64

	for _, destPath := range localFiles {
		if fi, err := os.Stat(destPath); err == nil {
			err = os.Remove(destPath)
			if err != nil {
				klog.Warningf("Failed to delete %q: %v", destPath, err)
			} else {
				deletedCount++
				deletedBytes += fi.Size()
				fmt.Printf("Deleted: %s\n", destPath)
			}
		}
	}

	if deletedCount > 0 {
		fmt.Printf("\nSuccessfully deleted %d files (freed %s).\n",
			deletedCount,
			humanize.Bytes(uint64(deletedBytes)),
		)
	} else {
		fmt.Println("No local cache files were deleted.")
	}
}

func listDownloadedFiles(ds *datasets.Dataset, config, split string) {
	fmt.Printf("Locating local cached files for dataset %q, config %q, split %q...\n",
		ds.ID,
		cmp.Or(config, "*"),
		cmp.Or(split, "*"),
	)

	localFiles, err := ds.ListDownloadedFiles(config, split)
	if err != nil {
		klog.Fatalf("Error listing local files: %+v", err)
	}

	if len(localFiles) == 0 {
		fmt.Println("No local cache files found matching the criteria.")
		return
	}

	fmt.Printf("\nDownloaded files:\n")
	var totalSize int64
	for _, destPath := range localFiles {
		sizeStr := "-"
		if fi, err := os.Stat(destPath); err == nil {
			sizeStr = humanize.Bytes(uint64(fi.Size()))
			totalSize += fi.Size()
		}
		fmt.Printf("  - %s (%s)\n", destPath, sizeStr)
	}
	fmt.Printf("\nTotal files: %d, total size: %s\n", len(localFiles), humanize.Bytes(uint64(totalSize)))
}

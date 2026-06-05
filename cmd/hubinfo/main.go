package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/lipgloss/table"
	"github.com/gomlx/compute/support/humanize"
	"github.com/gomlx/go-huggingface/hub"
)

var (
	fullFlag = flag.Bool("full", false, "list all available repository metadata")
)

func main() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] <repo_id>\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Example: %s Qwen/Qwen3-0.6B\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Flags:\n")
		flag.PrintDefaults()
	}
	flag.Parse()

	if flag.NArg() != 1 {
		flag.Usage()
		os.Exit(1)
	}

	repoID := flag.Arg(0)

	// Fetch repository info
	repo := hub.New(repoID)
	err := repo.DownloadInfo(false)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error downloading repository info: %+v\n", err)
		os.Exit(1)
	}

	info := repo.Info()
	if info == nil {
		fmt.Fprintf(os.Stderr, "Error: repository info is empty\n")
		os.Exit(1)
	}

	// Prepare Metadata Table
	metaTable := table.New().
		Border(lipgloss.NormalBorder()).
		BorderStyle(lipgloss.NewStyle().Foreground(lipgloss.Color("238"))).
		Headers("Metadata Field", "Value")

	baseStyle := lipgloss.NewStyle().Padding(0, 1)
	metaTable.StyleFunc(func(row, col int) lipgloss.Style {
		s := baseStyle
		if row == table.HeaderRow {
			return s.Foreground(lipgloss.Color("252")).Bold(true)
		}
		if col == 0 {
			return s.Bold(true).Foreground(lipgloss.Color("86")) // Cyan-like color for keys
		}
		if col == 1 {
			return s.Width(80) // Wrap the value column if it gets too wide
		}
		return s
	})

	metaTable.Row("Model ID", repoID)
	metaTable.Row("Tags", strings.Join(info.Tags, ", "))
	metaTable.Row("Downloads / Likes", fmt.Sprintf("%s / %s", humanize.Count(info.Downloads), humanize.Count(info.Likes)))
	metaTable.Row("Latest Update", info.LastModified.Local().Format(time.RFC1123))

	if info.Disabled {
		redStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("9")).Bold(true)
		metaTable.Row("Disabled", redStyle.Render("YES"))
	}

	metaTable.Row("SHA", info.CommitHash)

	if info.UsedStorage > 0 {
		metaTable.Row("Used Storage", humanize.Bytes(info.UsedStorage))
	}

	if *fullFlag {
		metaTable.Row("Private", fmt.Sprintf("%t", info.Private))
		if info.LibraryName != "" {
			metaTable.Row("Library Name", info.LibraryName)
		}
		if info.PipelineTag != "" {
			metaTable.Row("Pipeline Tag", info.PipelineTag)
		}
		if info.Gated != nil {
			metaTable.Row("Gated", fmt.Sprintf("%v", info.Gated))
		}
		if !info.CreatedAt.IsZero() {
			metaTable.Row("Created At", info.CreatedAt.Local().Format(time.RFC1123))
		}
		if len(info.Spaces) > 0 {
			metaTable.Row("Spaces", strings.Join(info.Spaces, ", "))
		}
		if info.Config != nil {
			if len(info.Config.Architectures) > 0 {
				metaTable.Row("Config: Architectures", strings.Join(info.Config.Architectures, ", "))
			}
			if info.Config.ModelType != "" {
				metaTable.Row("Config: Model Type", info.Config.ModelType)
			}
		}
		if info.CardData != nil {
			if info.CardData.License != "" {
				metaTable.Row("Card Data: License", info.CardData.License)
			}
			if info.CardData.LicenseLink != "" {
				metaTable.Row("Card Data: License Link", info.CardData.LicenseLink)
			}
			if info.CardData.BaseModel != nil {
				metaTable.Row("Card Data: Base Model", fmt.Sprintf("%v", info.CardData.BaseModel))
			}
		}
		if info.Transformers != nil {
			if info.Transformers.AutoModel != "" {
				metaTable.Row("Transformers: Auto Model", info.Transformers.AutoModel)
			}
			if info.Transformers.Processor != "" {
				metaTable.Row("Transformers: Processor", info.Transformers.Processor)
			}
		}
	}

	fmt.Println(lipgloss.NewStyle().Foreground(lipgloss.Color("99")).Bold(true).Render("Repository Metadata:"))
	fmt.Println(metaTable)
	fmt.Println()

	// Prepare Files Table
	filesTable := table.New().
		Border(lipgloss.NormalBorder()).
		BorderStyle(lipgloss.NewStyle().Foreground(lipgloss.Color("238"))).
		Headers("File Name", "Size", "LFS")

	filesTable.StyleFunc(func(row, col int) lipgloss.Style {
		s := baseStyle
		if row == table.HeaderRow {
			return s.Foreground(lipgloss.Color("252")).Bold(true)
		}
		if col == 1 {
			return s.Align(lipgloss.Right) // right-align sizes
		}
		return s
	})

	for fi, err := range repo.IterFileInfos() {
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error iterating files: %v\n", err)
			os.Exit(1)
		}
		sizeStr := humanize.Bytes(fi.Size)
		lfsStr := "-"
		if fi.LFS != nil {
			lfsStr = "Yes"
		}
		filesTable.Row(fi.Name, sizeStr, lfsStr)
	}

	fmt.Println(lipgloss.NewStyle().Foreground(lipgloss.Color("99")).Bold(true).Render("Files:"))
	fmt.Println(filesTable)
}

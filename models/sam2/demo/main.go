// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	_ "image/jpeg"
	"image/png"
	_ "image/png"
	"log"
	"os"
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/models/sam2"
	_ "github.com/gomlx/gomlx/backends/default"
)

var (
	inputFlag     = flag.String("input", "", "Path to the input image file (JPEG or PNG)")
	outputFlag    = flag.String("output", "output.png", "Path to save the output segmented image")
	modelFlag     = flag.String("model", "facebook/sam2-hiera-base-plus", "HuggingFace repo ID of the SAM2 model")
	pointsFlag    = flag.String("points", "", "Points to segment in format 'x1,y1,label1;x2,y2,label2' (label: 1=fg, 0=bg)")
	boxesFlag     = flag.String("boxes", "", "Bounding boxes to segment in format 'x_min,y_min,x_max,y_max;...'")
	colorFlag     = flag.String("color", "gray", "Highlight color for the mask ('red', 'green', 'blue', 'gray', 'yellow' or hex '#RRGGBB' or 'R,G,B')")
	multimaskFlag = flag.Bool("multimask", false, "Output all 3 ambiguous masks (will save as <output>_0.png, <output>_1.png, <output>_2.png)")
	formatFlag    = flag.String("format", "", "Force output image format ('png' or 'jpg'), default infers from output file extension")
)

func main() {
	flag.Parse()

	if *inputFlag == "" {
		fmt.Println("Error: -input image flag is required.")
		flag.Usage()
		os.Exit(1)
	}

	if *pointsFlag == "" && *boxesFlag == "" {
		log.Fatalf("Error: at least one point prompt (-points) or box prompt (-boxes) must be provided.")
	}

	points, err := parsePoints(*pointsFlag)
	if err != nil {
		log.Fatalf("Failed to parse points: %v", err)
	}

	boxes, err := parseBoxes(*boxesFlag)
	if err != nil {
		log.Fatalf("Failed to parse boxes: %v", err)
	}

	maskColor := parseColor(*colorFlag)

	// 1. Load image
	img, err := loadImage(*inputFlag)
	if err != nil {
		log.Fatalf("Failed to load input image: %v", err)
	}

	// 2. Download and load SAM2 model
	fmt.Printf("Loading model info from HuggingFace for %s...\n", *modelFlag)
	repo := hub.New(*modelFlag)
	if err := repo.DownloadInfo(false); err != nil {
		log.Fatalf("Failed to download model info: %v", err)
	}

	fmt.Println("Loading model configuration and weights...")
	modelObj, err := sam2.LoadModel(repo)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	fmt.Println("Initializing GoMLX backend...")
	backend, err := compute.New()
	if err != nil {
		log.Fatalf("Failed to create backend: %v", err)
	}
	defer backend.Finalize()

	fmt.Println("Creating Segmenter...")
	segmenter, err := sam2.NewSegmenter(backend, modelObj)
	if err != nil {
		log.Fatalf("Failed to create segmenter: %v", err)
	}

	// 3. Segment image
	options := &sam2.SegmentationOptions{
		Points:          points,
		Boxes:           boxes,
		MultiMaskOutput: *multimaskFlag,
	}

	fmt.Println("Running inference on backend...")
	segmentations, err := segmenter.Segment(img, options)
	if err != nil {
		log.Fatalf("Segmentation failed: %v", err)
	}

	fmt.Printf("Inference completed. Found %d mask(s).\n", len(segmentations))

	// 4. Save output(s)
	if *multimaskFlag {
		for i, seg := range segmentations {
			outImg := overlayMask(img, seg.Mask, maskColor)
			outputPath := formatIndexedPath(*outputFlag, i)
			fmt.Printf("Saving mask %d (IoU score: %.4f) to %s...\n", i, seg.IoUScore, outputPath)
			if err := saveImage(outImg, outputPath, *formatFlag); err != nil {
				log.Fatalf("Failed to save output image: %v", err)
			}
		}
	} else {
		// Just save the best one (the list is already sorted descending by IoU)
		seg := segmentations[0]
		outImg := overlayMask(img, seg.Mask, maskColor)
		fmt.Printf("Saving best mask (IoU score: %.4f) to %s...\n", seg.IoUScore, *outputFlag)
		if err := saveImage(outImg, *outputFlag, *formatFlag); err != nil {
			log.Fatalf("Failed to save output image: %v", err)
		}
	}
	fmt.Println("Done!")
}

func loadImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	return img, err
}

func saveImage(img image.Image, path string, format string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	format = strings.ToLower(format)
	if format == "" {
		if strings.HasSuffix(strings.ToLower(path), ".jpg") || strings.HasSuffix(strings.ToLower(path), ".jpeg") {
			format = "jpg"
		} else {
			format = "png"
		}
	}

	if format == "jpg" || format == "jpeg" {
		return jpeg.Encode(f, img, &jpeg.Options{Quality: 90})
	}
	return png.Encode(f, img)
}

func formatIndexedPath(originalPath string, index int) string {
	extIdx := strings.LastIndex(originalPath, ".")
	if extIdx == -1 {
		return fmt.Sprintf("%s_%d", originalPath, index)
	}
	return fmt.Sprintf("%s_%d%s", originalPath[:extIdx], index, originalPath[extIdx:])
}

func parseColor(s string) color.RGBA {
	s = strings.ToLower(strings.TrimSpace(s))
	switch s {
	case "red":
		return color.RGBA{R: 255, G: 0, B: 0, A: 255}
	case "green":
		return color.RGBA{R: 0, G: 255, B: 0, A: 255}
	case "blue":
		return color.RGBA{R: 0, G: 0, B: 255, A: 255}
	case "yellow":
		return color.RGBA{R: 255, G: 255, B: 0, A: 255}
	case "magenta":
		return color.RGBA{R: 255, G: 0, B: 255, A: 255}
	case "cyan":
		return color.RGBA{R: 0, G: 255, B: 255, A: 255}
	case "gray", "grey":
		return color.RGBA{R: 128, G: 128, B: 128, A: 255}
	}

	if strings.HasPrefix(s, "#") {
		var r, g, b uint8
		n, _ := fmt.Sscanf(s, "#%02x%02x%02x", &r, &g, &b)
		if n == 3 {
			return color.RGBA{R: r, G: g, B: b, A: 255}
		}
	}

	var r, g, b int
	n, _ := fmt.Sscanf(s, "%d,%d,%d", &r, &g, &b)
	if n == 3 {
		return color.RGBA{R: uint8(r), G: uint8(g), B: uint8(b), A: 255}
	}

	return color.RGBA{R: 128, G: 128, B: 128, A: 255}
}

func parsePoints(s string) ([]sam2.PromptPoint, error) {
	if s == "" {
		return nil, nil
	}
	var points []sam2.PromptPoint
	parts := strings.Split(s, ";")
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		var x, y, labelVal int
		_, err := fmt.Sscanf(part, "%d,%d,%d", &x, &y, &labelVal)
		if err != nil {
			return nil, fmt.Errorf("invalid point format %q, expected x,y,label", part)
		}
		points = append(points, sam2.PromptPoint{
			X:     x,
			Y:     y,
			Label: sam2.PointLabel(labelVal),
		})
	}
	return points, nil
}

func parseBoxes(s string) ([]sam2.PromptBox, error) {
	if s == "" {
		return nil, nil
	}
	var boxes []sam2.PromptBox
	parts := strings.Split(s, ";")
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		var minX, minY, maxX, maxY int
		_, err := fmt.Sscanf(part, "%d,%d,%d,%d", &minX, &minY, &maxX, &maxY)
		if err != nil {
			return nil, fmt.Errorf("invalid box format %q, expected x_min,y_min,x_max,y_max", part)
		}
		boxes = append(boxes, sam2.PromptBox{
			MinX: minX,
			MinY: minY,
			MaxX: maxX,
			MaxY: maxY,
		})
	}
	return boxes, nil
}

func overlayMask(orig image.Image, mask image.Image, maskColor color.RGBA) image.Image {
	bounds := orig.Bounds()
	out := image.NewRGBA(bounds)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			origColor := orig.At(x, y)
			r, g, b, a := origColor.RGBA()
			r8 := uint8(r >> 8)
			g8 := uint8(g >> 8)
			b8 := uint8(b >> 8)
			a8 := uint8(a >> 8)

			maskVal := mask.At(x-bounds.Min.X, y-bounds.Min.Y).(color.Gray).Y
			if maskVal > 0 {
				rBlend := uint8((uint16(r8) + uint16(maskColor.R)) / 2)
				gBlend := uint8((uint16(g8) + uint16(maskColor.G)) / 2)
				bBlend := uint8((uint16(b8) + uint16(maskColor.B)) / 2)
				out.Set(x, y, color.RGBA{R: rBlend, G: gBlend, B: bBlend, A: a8})
			} else {
				out.Set(x, y, color.RGBA{R: r8, G: g8, B: b8, A: a8})
			}
		}
	}
	return out
}

# SAM2: Segment Anything Model 2 in GoMLX

This package implements the image segmentation part of Facebook's **Segment Anything in Images And Videos (SAM2)** foundation model. It is designed to run efficiently in Go using [GoMLX](https://github.com/gomlx/gomlx) for hardware-accelerated tensor computations.

This package implements the image part of the model (including the Hiera vision backbone, neck, prompt encoder, and mask decoder), not the video tracking part.

## References
* **Paper**: [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/pdf/2408.00714)
* **Model Checkpoint**: [facebook/sam2-hiera-base-plus on HuggingFace](https://huggingface.co/facebook/sam2-hiera-base-plus)
* **Reference PyTorch Implementation**: [HuggingFace Transformers SAM2](https://github.com/huggingface/transformers/tree/main/src/transformers/models/sam2)

---

## APIs Offered

The package exposes two levels of APIs:

### 1. High-Level Go API (Inference Segmenter)
A user-friendly, standard Go interface that operates on `image.Image` inputs and handles raw tensor conversion, preprocessing, and output resizing automatically.
* **[NewSegmenter](file:///home/janpf/Projects/gomlx/go-huggingface/models/sam2/goapi.go#L58)**: Initializes a predictor using the loaded config/weights and compiles the computation graphs.
* **[Segment](file:///home/janpf/Projects/gomlx/go-huggingface/models/sam2/goapi.go#L105)**: Predicts masks for a given image and prompt options.

### 2. Low-Level Graph API (GoMLX Graph Construction)
A graph-building API designed for users who want to embed SAM2 inside their custom GoMLX computational graphs, training loops, or pipelines.
* **[Forward](file:///home/janpf/Projects/gomlx/go-huggingface/models/sam2/sam2.go#L841)**: Constructs the full SAM2 graph (Vision Encoder $\rightarrow$ Prompt Encoder $\rightarrow$ Mask Decoder) using GoMLX `*Node` operations.
* **[InternalStates](file:///home/janpf/Projects/gomlx/go-huggingface/models/sam2/sam2.go#L801)**: A struct exposing intermediate outputs (e.g. FPN neck features, prompt embeddings, upscaled mask maps) for transfer learning and debugging.

---

## Code Structure
* [sam2.go](file:///home/janpf/Projects/gomlx/go-huggingface/models/sam2/sam2.go): The core model architecture (backbone, neck, attention layers, and mask decoder).
* [goapi.go](file:///home/janpf/Projects/gomlx/go-huggingface/models/sam2/goapi.go): The standard Go inference API wrapping the GoMLX executable.
* [model.go](file:///home/janpf/Projects/gomlx/go-huggingface/models/sam2/model.go): SafeTensors weights loading, variable mapping, and transpositions.
* [config.go](file:///home/janpf/Projects/gomlx/go-huggingface/models/sam2/config.go): Configuration parsing and model parameters.

---

## Usage Examples

### Using the High-Level `Segmenter`

```go
package main

import (
	"fmt"
	"image"
	_ "image/png"
	"log"
	"os"

	"github.com/gomlx/compute"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/models/sam2"
	_ "github.com/gomlx/gomlx/backends/default"
)

func main() {
	// Initialize GoMLX backend (e.g., CUDA, CPU)
	backend, err := compute.New()
	if err != nil {
		log.Fatalf("Failed to initialize backend: %v", err)
	}
	defer backend.Finalize()

	// Load configuration and weights from HuggingFace
	repo := hub.New("facebook/sam2-hiera-base-plus")
	modelObj, err := sam2.LoadModel(repo)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	// Create standard Segmenter
	segmenter, err := sam2.NewSegmenter(backend, modelObj)
	if err != nil {
		log.Fatalf("Failed to create segmenter: %v", err)
	}

	// Load target image
	imgFile, err := os.Open("target.png")
	if err != nil {
		log.Fatalf("Failed to open image: %v", err)
	}
	defer imgFile.Close()
	img, _, err := image.Decode(imgFile)

	// Set point prompt (foreground point at x=512, y=512)
	options := &sam2.PredictOptions{
		Points: []sam2.PromptPoint{
			{X: 512, Y: 512, Label: sam2.LabelForeground},
		},
		MultiMaskOutput: false,
	}

	// Segment image
	segmentations, err := segmenter.Segment(img, options)
	if err != nil {
		log.Fatalf("Segmentation failed: %v", err)
	}

	best := segmentations[0]
	fmt.Printf("Predicted mask with IoU score: %.4f\n", best.IoUScore)
	// best.Mask contains the grey/binary mask image.Image
}
```

---

## Demo CLI Program

A command-line program is available under [models/sam2/demo](file:///home/janpf/Projects/gomlx/go-huggingface/models/sam2/demo/main.go) to segment arbitrary images.

### Running the Demo
1. Build the demo binary:
   ```bash
   go build -o sam2-demo ./models/sam2/demo
   ```

2. Segment an image using a point prompt (`-points "x,y,label"` where label `1` is foreground, `0` is background) and overlay a red highlight mask:
   ```bash
   ./sam2-demo -input input.png -output output.png -points "512,512,1" -color "red"
   ```

3. Segment using a bounding box prompt (`-boxes "x_min,y_min,x_max,y_max"`):
   ```bash
   ./sam2-demo -input input.png -output output.png -boxes "100,100,800,800" -color "blue"
   ```

4. Run in multi-mask mode to output all 3 candidate masks:
   ```bash
   ./sam2-demo -input input.png -output output.png -points "512,512,1" -multimask=true
   ```

### Demo Flags:
* `-input`: Path to the input image file (JPEG or PNG).
* `-output`: Path to save the output segmented image (defaults to `output.png`).
* `-model`: HuggingFace repository ID of the model (defaults to `facebook/sam2-hiera-base-plus`).
* `-points`: Semicolon-separated point coordinates (e.g. `x1,y1,label1;x2,y2,label2`).
* `-boxes`: Semicolon-separated box coordinates (e.g. `xmin,ymin,xmax,ymax`).
* `-color`: Mask overlay highlight color (supports hex like `#ff0000`, RGB like `255,0,0`, or names like `red`, `green`, `blue`, `gray`, `yellow`).
* `-multimask`: Output 3 ambiguous masks as separate files (`output_0.png`, `output_1.png`, etc.).
* `-format`: Output image format (`png` or `jpg`).

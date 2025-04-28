# partition_pdf function

## Resulting Element Types After Chunking

### `CompositeElement`
A `CompositeElement` is any text element that results from chunking. It can be a combination of multiple original text elements that fit within the chunk size, a single element that fits without allowing others, or a fragment of a larger element that required splitting.

### `Table`
A `Table` element represents a structured table extracted from the document. It is never combined with other elements, and if it fits within the `max_characters` limit, it remains unchanged.

### `TableChunk`
A `TableChunk` is a portion of a large table that exceeds the `max_characters` limit. When a table is too big to fit in one chunk, it is split into multiple `TableChunk` elements while preserving its structure.

## Parameters

### Chunking Strategies

#### 1. `basic`
The **basic** chunking strategy combines sequential elements to maximally fill each chunk while respecting both the specified `max_characters` (hard-max) and `new_after_n_chars` (soft-max) option values.

- A single element that exceeds the hard-max is isolated (never combined with another element) and then divided into two or more chunks using text-splitting.
- A **Table** element is always isolated and never combined with another element. If a table exceeds the hard-max, it is divided into multiple `TableChunk` elements using text-splitting.
- If specified, overlap is applied between chunks formed by splitting oversized elements and is also applied between other chunks when `overlap_all` is True.

**Best for:**
- Documents with relatively straightforward content where chunks should be filled efficiently while preserving element boundaries.

---

#### 2. `by_title`
The **by_title** chunking strategy preserves section and optional page boundaries. A new chunk starts when a Title element or page break is encountered.

**Key Behaviors:**
- Detects section headings and starts a new chunk at each Title element.
- Optionally respects page breaks, separating content across pages into different chunks.
- Combines small sections if needed to fill chunks efficiently.

**Best for:**
- Documents where it's essential to maintain the structure of sections and pages, like reports or books.

---

### Partitioning Strategies

#### 1. `auto` (Default)
The “auto” strategy will choose the partitioning strategy based on document characteristics and the function kwargs.

**Best for:**
- General-purpose document processing with an adaptive approach.

---

#### 2. `fast`
The “rule-based” strategy leverages traditional NLP extraction techniques to quickly pull all the text elements. “Fast” strategy is not recommended for image-based file types.

**Best for:**
- Simple text-based PDFs with minimal formatting.

---

#### 3. `hi_res`
The “model-based” strategy identifies the layout of the document. The advantage of “hi_res” is that it uses the document layout to gain additional information about document elements. We recommend using this strategy if your use case is highly sensitive to correct classifications for document elements.

**Best for:**
- PDFs with complex layouts (e.g., reports, research papers).

---

#### 4. `ocr_only`
Another “model-based” strategy that leverages Optical Character Recognition to extract text from the image-based files.

**Best for:**
- Scanned documents and PDFs with embedded images.

---

#### 5. `vlm`
Uses a Vision-Language Model (VLM) to extract text from these file types: `.bmp, .gif, .heic, .jpeg, .jpg, .pdf, .png, .tiff, and .webp`.

**Best for:**
- Documents with a combination of text, images, and tables.
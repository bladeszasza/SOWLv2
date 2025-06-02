# SOWLv2 Architecture

## System Overview

```mermaid
graph TB
    subgraph Input
        I[Image/Video Input]
        P[Text Prompts]
    end

    subgraph Core Pipeline
        O[OWLv2 Detection]
        S[SAM2 Segmentation]
        T[Tracking]
    end

    subgraph Output
        B[Binary Masks]
        OV[Overlays]
        M[Merged Outputs]
        V[Video Outputs]
    end

    I --> O
    P --> O
    O --> S
    S --> T
    T --> B
    T --> OV
    B --> M
    OV --> M
    M --> V
```

## Component Interaction

```mermaid
sequenceDiagram
    participant CLI
    participant Pipeline
    participant OWL
    participant SAM
    participant Utils

    CLI->>Pipeline: process_image/process_video
    Pipeline->>OWL: detect(prompt, image)
    OWL-->>Pipeline: detections
    Pipeline->>SAM: segment(image, box)
    SAM-->>Pipeline: mask
    Pipeline->>Utils: create_outputs
    Utils-->>Pipeline: saved files
    Pipeline-->>CLI: results
```

## Class Structure

```mermaid
classDiagram
    class SOWLv2Pipeline {
        +process_image()
        +process_video()
        +process_frames()
        -_process_single_detection()
        -_save_detection_outputs()
    }

    class PipelineConfig {
        +binary: bool
        +overlay: bool
        +merged: bool
    }

    class VideoTrackingConfig {
        +prompt: Union[str, List[str]]
        +threshold: float
        +prompt_color_map: Dict
        +palette: List[Tuple]
        +next_color_idx: int
        +fps: int
    }

    class VideoProcessingConfig {
        +pipeline_config: PipelineConfig
        +prompt_color_map: Dict
        +next_color_idx: int
        +fps: int
    }

    SOWLv2Pipeline --> PipelineConfig
    SOWLv2Pipeline --> VideoTrackingConfig
    SOWLv2Pipeline --> VideoProcessingConfig
```

## Data Flow

```mermaid
flowchart LR
    subgraph Input Processing
        I[Input Image/Video]
        P[Text Prompts]
        D[Detection Boxes]
    end

    subgraph Mask Generation
        M[Masks]
        V[Validated Masks]
        C[Colored Overlays]
    end

    subgraph Output Generation
        B[Binary Masks]
        O[Overlays]
        F[Final Outputs]
    end

    I --> D
    P --> D
    D --> M
    M --> V
    V --> C
    V --> B
    C --> O
    B --> F
    O --> F
```

## Directory Structure

```mermaid
graph TD
    subgraph SOWLv2
        S[sowlv2/]
        S --> C[cli.py]
        S --> P[pipeline.py]
        S --> I[image_pipeline.py]
        S --> V[video_pipeline.py]
        S --> F[frame_pipeline.py]
        S --> U[pipeline_utils.py]
        S --> D[data/]
        S --> M[models/]
    end

    subgraph Data
        D --> DC[config.py]
        D --> DT[types.py]
    end

    subgraph Models
        M --> MO[owl.py]
        M --> MS[sam2_wrapper.py]
    end
```

## Error Handling Flow

```mermaid
graph TD
    subgraph Error Handling
        E[Error Occurs]
        E --> T{Error Type}
        T -->|IOError| I[File Operations]
        T -->|ValueError| V[Data Validation]
        T -->|RuntimeError| R[Processing]
        T -->|OSError| O[System]
        I --> H[Handle & Log]
        V --> H
        R --> H
        O --> H
        H --> C[Continue/Exit]
    end
```

## Configuration Flow

```mermaid
graph LR
    subgraph Configuration
        D[Default Config]
        U[User Config]
        C[CLI Args]
        D --> M[Merged Config]
        U --> M
        C --> M
        M --> P[Pipeline Config]
        M --> V[Video Config]
        M --> I[Image Config]
    end
```

These diagrams provide a comprehensive view of the SOWLv2 architecture, including:

1. System overview and component interaction
2. Class structure and relationships
3. Data flow through the system
4. Directory structure
5. Error handling flow
6. Configuration management

Each diagram focuses on a different aspect of the system, making it easier to understand the overall architecture and how different components interact with each other.

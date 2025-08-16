# Features

- Easier setup and quickstart
- Built-in CLI and Python API
- Works with arXiv links and local PDFs
- Configurable output formats
- Extensible pipeline

## Diagrams

```mermaid
graph TD;
  A[Paper] -->|extracts| B(Text + Figures);
  B --> C{Model Reader};
  C -->|predicts| D[Super-Weights];
```

## Tasks

- [x] GitHub Pages deployment
- [x] Material theme enhancements
- [ ] Dataset examples
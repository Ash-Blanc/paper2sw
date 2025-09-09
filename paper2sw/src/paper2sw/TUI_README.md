# Paper2SW TUI

The Paper2SW TUI (Text-based User Interface) provides an interactive terminal-based interface for the Paper2SW super-weight prediction tool.

## Features

- **Interactive Paper Input**: Enter paper URLs or file paths directly or browse for files
- **Real-time Predictions**: Get super-weight predictions with progress indicators
- **Result Filtering**: Filter predictions by layer, row, column, value, or model family
- **Data Visualization**: View prediction value distributions with sparklines
- **Export Functionality**: Export predictions to JSONL files
- **Configuration Management**: Adjust settings like top_k, keep_ratio, and model_id
- **Dark/Light Mode**: Toggle between dark and light color schemes

## Usage

To run the TUI, use the following command:

```bash
paper2sw tui
```

Or with specific options:

```bash
paper2sw tui --top_k 10 --keep_ratio 0.5 --no_cache
```

## Key Bindings

- `q` - Quit the application
- `r` - Reset the application
- `d` - Toggle dark mode

## Interface Components

### Paper Input Panel
- Enter paper URLs or file paths
- Browse for files using the file browser
- Adjust prediction settings

### Model Information Panel
- View model family and status
- Monitor prediction progress

### Results Panel
- View prediction results in a sortable table
- Filter results by any field
- Visualize value distributions
- Export predictions to files

## Development

The TUI is built using the [Textual](https://textual.textualize.io/) framework. To modify or extend the interface:

1. Edit `tui.py` for main application logic
2. Modify `tui.tcss` for styling
3. Update `file_browser.py` for file browsing functionality
4. Modify `settings.py` for settings management

## Screenshots

*(Screenshots would be added here showing the TUI interface)*
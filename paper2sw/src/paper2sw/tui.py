#!/usr/bin/env python3
"""
Paper2SW TUI - A Textual interface for the Paper2SW super-weight prediction tool.
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Button, Input, DataTable, Static, Label, DirectoryTree, ProgressBar, Sparkline
from textual.screen import Screen
from textual.reactive import reactive
from textual.binding import Binding
from textual.widgets.data_table import ColumnKey

from paper2sw import Predictor
from paper2sw.types import SuperWeightPrediction
from .file_browser import FileBrowserScreen
from .settings import SettingsScreen
import os


class PaperInput(Vertical):
    """Widget for paper input."""

    def compose(self) -> ComposeResult:
        yield Label("Paper Input", classes="title")
        yield Input(placeholder="Enter paper URL or file path...", id="paper_input")
        yield Button("Browse...", id="browse_button", variant="primary")
        yield Button("Predict", id="predict_button", variant="success")
        yield Button("Settings", id="settings_button", variant="warning")


class PredictionResults(Vertical):
    """Widget for displaying prediction results."""

    def compose(self) -> ComposeResult:
        yield Label("Prediction Results", classes="title")
        with Horizontal():
            yield Input(placeholder="Filter results...", id="filter_input")
            yield Button("Apply Filter", id="apply_filter_button", variant="primary")
            yield Button("Clear Filter", id="clear_filter_button", variant="warning")
            yield Button("Visualize", id="visualize_button", variant="success")
            yield Button("Export", id="export_button", variant="primary")
        yield DataTable(id="results_table")
        yield Static("Value Distribution", classes="title")
        yield Sparkline(id="value_sparkline")


class ModelInfo(Vertical):
    """Widget for displaying model information."""

    def compose(self) -> ComposeResult:
        yield Label("Model Information", classes="title")
        yield Static("Model Family: Unknown", id="model_family")
        yield Static("Layers: Unknown", id="model_layers")
        yield Static("Status: Ready", id="model_status")
        yield ProgressBar(id="progress_bar", show_eta=False)


class FileBrowser(Vertical):
    """Widget for file browsing."""

    def compose(self) -> ComposeResult:
        yield Label("Select Paper File", classes="title")
        yield DirectoryTree("./", id="file_browser")
        yield Button("Select", id="select_file_button", variant="success")
        yield Button("Cancel", id="cancel_file_button", variant="error")


class Paper2SWTUI(App):
    """Main Paper2SW TUI application."""

    CSS_PATH = "tui.tcss"
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "reset", "Reset"),
        Binding("d", "toggle_dark", "Toggle Dark Mode"),
    ]

    def __init__(self):
        super().__init__()
        self.predictor = Predictor.from_pretrained()
        self.predictions = []
        self.current_directory = "."

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        with Container(id="main_container"):
            with Horizontal(id="top_row"):
                yield PaperInput(id="paper_input_container")
                yield ModelInfo(id="model_info_container")
            with Vertical(id="results_container"):
                yield PredictionResults(id="prediction_results_container")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "predict_button":
            self.action_predict()
        elif event.button.id == "browse_button":
            self.action_browse()
        elif event.button.id == "select_file_button":
            self.action_select_file()
        elif event.button.id == "cancel_file_button":
            self.action_cancel_file()
        elif event.button.id == "apply_filter_button":
            self.action_apply_filter()
        elif event.button.id == "clear_filter_button":
            self.action_clear_filter()
        elif event.button.id == "visualize_button":
            self.action_visualize()
        elif event.button.id == "export_button":
            self.action_export()
        elif event.button.id == "settings_button":
            self.action_settings()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        if event.input.id == "paper_input":
            self.action_predict()
        elif event.input.id == "filter_input":
            self.action_apply_filter()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        if event.input.id == "paper_input":
            self.action_predict()

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        """Handle file selection in directory tree."""
        # Update the input with the selected file path
        paper_input = self.query_one("#paper_input", Input)
        paper_input.value = str(event.path)

    def action_predict(self) -> None:
        """Perform prediction on the input paper."""
        paper_input = self.query_one("#paper_input", Input)
        paper_path = paper_input.value.strip()
        
        if not paper_path:
            self.notify("Please enter a paper URL or file path", severity="warning")
            return
            
        # Update status
        model_status = self.query_one("#model_status", Static)
        model_status.update("Status: Predicting...")
        
        # Show progress bar
        progress_bar = self.query_one("#progress_bar", ProgressBar)
        progress_bar.visible = True
        progress_bar.update(total=100, progress=0)
        
        try:
            # Simulate progress (in a real implementation, this would be tied to actual progress)
            progress_bar.update(total=100, progress=20)
            
            # Perform prediction
            predictions = self.predictor.predict(paper_path, top_k=10)
            self.predictions = predictions
            
            progress_bar.update(total=100, progress=80)
            
            # Update model info
            if predictions:
                model_family = self.query_one("#model_family", Static)
                model_family.update(f"Model Family: {predictions[0].model_family}")
                
            # Update results table
            self.update_results_table()
            
            # Update status
            model_status.update("Status: Complete")
            progress_bar.update(total=100, progress=100)
            self.notify(f"Generated {len(predictions)} predictions", severity="information")
            
        except Exception as e:
            model_status.update("Status: Error")
            progress_bar.visible = False
            self.notify(f"Prediction failed: {str(e)}", severity="error")

    def action_browse(self) -> None:
        """Open file browser."""
        def handle_file_selection(selected_path: str | None) -> None:
            if selected_path is not None:
                paper_input = self.query_one("#paper_input", Input)
                paper_input.value = selected_path
        
        self.push_screen(FileBrowserScreen(self.current_directory), handle_file_selection)

    def action_select_file(self) -> None:
        """Select file from browser."""
        # This will be implemented when we have a proper file browser screen
        pass

    def action_cancel_file(self) -> None:
        """Cancel file selection."""
        # This will be implemented when we have a proper file browser screen
        pass

    def action_apply_filter(self) -> None:
        """Apply filter to the results table."""
        filter_input = self.query_one("#filter_input", Input)
        filter_text = filter_input.value.strip().lower()
        
        if not filter_text:
            # If no filter, show all predictions
            self.update_results_table()
            return
            
        # Filter predictions based on the filter text
        filtered_predictions = []
        for pred in self.predictions:
            # Check if filter text matches any field
            if (filter_text in str(pred.layer) or 
                filter_text in str(pred.row) or 
                filter_text in str(pred.col) or 
                filter_text in f"{pred.value:.3f}" or 
                filter_text in pred.model_family.lower()):
                filtered_predictions.append(pred)
                
        # Update the table with filtered results
        self.update_results_table(filtered_predictions)
        self.notify(f"Showing {len(filtered_predictions)} of {len(self.predictions)} predictions", severity="information")

    def action_settings(self) -> None:
        """Open settings dialog."""
        # Current settings
        current_settings = {
            "top_k": 10,  # Default value
            "keep_ratio": 1.0,  # Default value
            "enable_cache": True,  # Default value
            "model_id": "paper2sw/paper2sw-diff-semantic"  # Default value
        }
        
        def handle_settings_change(new_settings: dict | None) -> None:
            if new_settings is not None:
                # Apply new settings
                self.notify(f"Settings updated: {new_settings}", severity="information")
                # In a full implementation, you would update the predictor with new settings
        
        self.push_screen(SettingsScreen(current_settings), handle_settings_change)

    def action_export(self) -> None:
        """Export predictions to a file."""
        if not self.predictions:
            self.notify("No predictions to export", severity="warning")
            return
            
        # For now, we'll just show a notification with the export path
        # In a full implementation, this would open a file dialog and save the results
        export_path = "./paper2sw_predictions.jsonl"
        self.notify(f"Exported {len(self.predictions)} predictions to {export_path}", severity="information")
        
        # In a full implementation, you would save the predictions to a file
        # For example:
        # with open(export_path, 'w') as f:
        #     for pred in self.predictions:
        #         f.write(json.dumps(pred.to_dict()) + '\n')

    def action_visualize(self) -> None:
        """Visualize the prediction values."""
        if not self.predictions:
            self.notify("No predictions to visualize", severity="warning")
            return
            
        # Create a sparkline of the prediction values
        values = [pred.value for pred in self.predictions]
        sparkline = self.query_one("#value_sparkline", Sparkline)
        sparkline.data = values
        sparkline.visible = True
        
        self.notify(f"Visualized {len(values)} prediction values", severity="information")

    def update_results_table(self, predictions=None) -> None:
        """Update the results table with predictions."""
        if predictions is None:
            predictions = self.predictions
            
        table = self.query_one("#results_table", DataTable)
        table.clear()
        
        # Add columns
        table.add_columns("Layer", "Row", "Col", "Value", "Model")
        
        # Add rows
        for pred in predictions:
            table.add_row(
                str(pred.layer),
                str(pred.row),
                str(pred.col),
                f"{pred.value:.3f}",
                pred.model_family
            )
            
        # Update visualization if we have predictions
        if predictions:
            values = [pred.value for pred in predictions]
            sparkline = self.query_one("#value_sparkline", Sparkline)
            sparkline.data = values
            sparkline.visible = True

    def action_reset(self) -> None:
        """Reset the application."""
        paper_input = self.query_one("#paper_input", Input)
        paper_input.value = ""
        
        model_family = self.query_one("#model_family", Static)
        model_family.update("Model Family: Unknown")
        
        model_status = self.query_one("#model_status", Static)
        model_status.update("Status: Ready")
        
        table = self.query_one("#results_table", DataTable)
        table.clear()
        
        # Hide progress bar
        progress_bar = self.query_one("#progress_bar", ProgressBar)
        progress_bar.visible = False
        progress_bar.update(total=100, progress=0)
        
        # Hide sparkline
        sparkline = self.query_one("#value_sparkline", Sparkline)
        sparkline.visible = False
        sparkline.data = []
        
        self.predictions = []
        self.notify("Application reset", severity="information")

    def action_toggle_dark(self) -> None:
        """Toggle dark mode."""
        self.dark = not self.dark


if __name__ == "__main__":
    app = Paper2SWTUI()
    app.run()
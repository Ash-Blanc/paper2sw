from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Input, Checkbox


class SettingsScreen(ModalScreen[dict]):
    """Modal screen for application settings."""

    def __init__(self, settings: dict) -> None:
        super().__init__()
        self.settings = settings

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label("Settings", classes="title")
            
            with Vertical(id="settings_form"):
                yield Label("Top K:")
                yield Input(
                    placeholder="Number of predictions",
                    id="top_k_input",
                    value=str(self.settings.get("top_k", 5))
                )
                
                yield Label("Keep Ratio:")
                yield Input(
                    placeholder="Text keep ratio (0.0-1.0)",
                    id="keep_ratio_input",
                    value=str(self.settings.get("keep_ratio", 1.0))
                )
                
                yield Checkbox(
                    "Enable Cache",
                    id="cache_checkbox",
                    value=self.settings.get("enable_cache", True)
                )
                
                yield Label("Model ID:")
                yield Input(
                    placeholder="Model identifier",
                    id="model_id_input",
                    value=self.settings.get("model_id", "paper2sw/paper2sw-diff-semantic")
                )
            
            with Container(id="buttons"):
                yield Button("Save", variant="primary", id="save")
                yield Button("Cancel", variant="error", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "save":
            # Collect settings from inputs
            try:
                top_k = int(self.query_one("#top_k_input", Input).value or "5")
                keep_ratio = float(self.query_one("#keep_ratio_input", Input).value or "1.0")
                enable_cache = self.query_one("#cache_checkbox", Checkbox).value
                model_id = self.query_one("#model_id_input", Input).value or "paper2sw/paper2sw-diff-semantic"
                
                settings = {
                    "top_k": top_k,
                    "keep_ratio": keep_ratio,
                    "enable_cache": enable_cache,
                    "model_id": model_id
                }
                self.dismiss(settings)
            except ValueError:
                self.notify("Invalid input values", severity="error")
        elif event.button.id == "cancel":
            self.dismiss(None)
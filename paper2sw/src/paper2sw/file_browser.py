from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Label


class FileBrowserScreen(ModalScreen[str]):
    """Modal screen for file browsing."""

    def __init__(self, current_directory: str = ".") -> None:
        super().__init__()
        self.current_directory = current_directory

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label("Select a paper file", classes="title")
            yield DirectoryTree(self.current_directory, id="file_tree")
            with Container(id="buttons"):
                yield Button("Select", variant="primary", id="select")
                yield Button("Cancel", variant="error", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "select":
            tree = self.query_one("#file_tree", DirectoryTree)
            if tree.cursor_path is not None:
                self.dismiss(str(tree.cursor_path))
        elif event.button.id == "cancel":
            self.dismiss(None)

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        """Handle file selection."""
        self.dismiss(str(event.path))
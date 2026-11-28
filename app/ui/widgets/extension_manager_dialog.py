"""Extension Manager dialog: browse a GitHub extensions repo and install/update/remove extensions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from app.helpers import extension_repo as repo

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

# Fetch threads that outlived their dialog; kept referenced until they finish.
_ORPHAN_THREADS: set["_FetchThread"] = set()

_STATE_TEXT = {
    repo.STATE_NOT_INSTALLED: "Not installed",
    repo.STATE_INSTALLED: "Installed",
    repo.STATE_UPDATE: "Update available",
}
_STATE_COLOR = {
    repo.STATE_INSTALLED: QtGui.QColor("#4caf50"),
    repo.STATE_UPDATE: QtGui.QColor("#ff9800"),
}


class _FetchThread(QtCore.QThread):
    loaded = QtCore.Signal(object)  # list[RepoExtension]
    failed = QtCore.Signal(str)

    def __init__(self, spec: repo.RepoSpec, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self._spec = spec

    def run(self) -> None:
        try:
            data = repo.download_repo_zip(self._spec)
            self.loaded.emit(repo.list_extensions(data, self._spec))
        except repo.ExtensionRepoError as e:
            self.failed.emit(str(e))
        except Exception as e:  # noqa: BLE001 - surface anything unexpected to the user
            self.failed.emit(f"Unexpected error: {e}")


class ExtensionManagerDialog(QtWidgets.QDialog):
    def __init__(self, main_window: "MainWindow", extensions_dir: Path):
        super().__init__(main_window)
        self.main_window = main_window
        self.extensions_dir = Path(extensions_dir)
        self._extensions: list[repo.RepoExtension] = []
        self._spec: Optional[repo.RepoSpec] = None
        self._thread: Optional[_FetchThread] = None
        self._restart_needed = False
        self._first_show = True

        self.setWindowTitle("Extension Manager")
        self.setWindowIcon(QtGui.QIcon(":/media/media/visomaster_small.png"))
        self.resize(900, 560)
        self._build_ui()
        self.repo_edit.setText(repo.load_repo_url(self.extensions_dir))
        self._update_buttons()

    # ----------------------------------------------------------------------- UI setup
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        repo_row = QtWidgets.QHBoxLayout()
        repo_row.addWidget(QtWidgets.QLabel("Repository:"))
        self.repo_edit = QtWidgets.QLineEdit()
        self.repo_edit.setPlaceholderText(repo.DEFAULT_REPO_URL)
        self.repo_edit.setToolTip(
            "A GitHub repository with one folder per extension, e.g.\n"
            "https://github.com/owner/repo/tree/main/extensions\n"
            "https://github.com/owner/repo  (uses the 'extensions' folder)\n"
            "owner/repo"
        )
        self.repo_edit.returnPressed.connect(self.refresh)
        repo_row.addWidget(self.repo_edit, 1)
        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        repo_row.addWidget(self.refresh_button)
        self.reset_button = QtWidgets.QPushButton("Default")
        self.reset_button.setToolTip("Restore the default repository")
        self.reset_button.clicked.connect(self._reset_repo)
        repo_row.addWidget(self.reset_button)
        layout.addLayout(repo_row)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Extension", "Status", "Description"])
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setWordWrap(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.itemDoubleClicked.connect(lambda *_: self._install_selected())
        splitter.addWidget(self.table)

        self.details = QtWidgets.QTextBrowser()
        self.details.setOpenExternalLinks(True)
        self.details.setPlaceholderText("Select an extension to see its details.")
        splitter.addWidget(self.details)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(6)
        self.progress.hide()
        layout.addWidget(self.progress)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.restart_label = QtWidgets.QLabel(
            "Restart VisoMaster for installed, updated or removed extensions to take effect."
        )
        self.restart_label.setStyleSheet("color: #ff9800;")
        self.restart_label.hide()
        layout.addWidget(self.restart_label)

        buttons = QtWidgets.QHBoxLayout()
        self.install_button = QtWidgets.QPushButton("Install")
        self.install_button.clicked.connect(self._install_selected)
        self.uninstall_button = QtWidgets.QPushButton("Uninstall")
        self.uninstall_button.clicked.connect(self._uninstall_selected)
        self.web_button = QtWidgets.QPushButton("View on GitHub")
        self.web_button.clicked.connect(self._open_web)
        self.folder_button = QtWidgets.QPushButton("Open Extensions Folder")
        self.folder_button.clicked.connect(self._open_folder)
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        for button in (self.install_button, self.uninstall_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        for button in (self.web_button, self.folder_button, close_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)

    # ------------------------------------------------------------------------ events
    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        if self._first_show:
            self._first_show = False
            QtCore.QTimer.singleShot(0, self.refresh)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self._retire_thread()
        super().closeEvent(event)

    # ---------------------------------------------------------------------- fetching
    def _retire_thread(self) -> None:
        thread = self._thread
        self._thread = None
        if thread is None:
            return
        try:
            thread.loaded.disconnect()
            thread.failed.disconnect()
        except (RuntimeError, TypeError):
            pass
        if thread.isRunning():
            _ORPHAN_THREADS.add(thread)
            thread.setParent(None)
            thread.finished.connect(lambda t=thread: _ORPHAN_THREADS.discard(t))
        else:
            thread.deleteLater()

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self.progress.setVisible(busy)
        self.refresh_button.setEnabled(not busy)
        self.repo_edit.setEnabled(not busy)
        self.reset_button.setEnabled(not busy)
        if message:
            self.status_label.setText(message)
        self._update_buttons()

    def refresh(self) -> None:
        try:
            spec = repo.parse_repo_url(self.repo_edit.text())
        except repo.ExtensionRepoError as e:
            self.status_label.setText(str(e))
            return

        self._retire_thread()
        self._spec = spec
        self._set_busy(True, f"Loading extensions from {spec.label} ...")
        thread = _FetchThread(spec, self)
        thread.loaded.connect(lambda exts, s=spec: self._on_loaded(exts, s))
        thread.failed.connect(self._on_failed)
        self._thread = thread
        thread.start()

    def _on_loaded(self, extensions: list, spec: repo.RepoSpec) -> None:
        self._extensions = extensions
        repo.save_repo_url(self.extensions_dir, self.repo_edit.text().strip())
        self._populate_table()
        count = len(extensions)
        if count:
            message = f"{count} extension{'s' if count != 1 else ''} available in {spec.label}."
        else:
            message = f"No extensions found in {spec.label}."
        self._set_busy(False, message)

    def _on_failed(self, message: str) -> None:
        self._extensions = []
        self._populate_table()
        self._set_busy(False, message)

    def _reset_repo(self) -> None:
        self.repo_edit.setText(repo.DEFAULT_REPO_URL)
        self.refresh()

    # ------------------------------------------------------------------------ table
    def _selected(self) -> Optional[repo.RepoExtension]:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        index = rows[0].row()
        return self._extensions[index] if 0 <= index < len(self._extensions) else None

    def _populate_table(self) -> None:
        selected = self._selected()
        selected_name = selected.name if selected else None
        self.table.blockSignals(True)
        self.table.setRowCount(len(self._extensions))
        for row, ext in enumerate(self._extensions):
            state = repo.installed_state(ext, self.extensions_dir)
            name_item = QtWidgets.QTableWidgetItem(ext.name)
            status_item = QtWidgets.QTableWidgetItem(_STATE_TEXT[state])
            if state in _STATE_COLOR:
                status_item.setForeground(QtGui.QBrush(_STATE_COLOR[state]))
            desc_item = QtWidgets.QTableWidgetItem(ext.description)
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, status_item)
            self.table.setItem(row, 2, desc_item)
        self.table.resizeRowsToContents()
        self.table.blockSignals(False)

        for row, ext in enumerate(self._extensions):
            if ext.name == selected_name:
                self.table.selectRow(row)
                break
        else:
            self.table.clearSelection()
        self._on_selection_changed()

    def _on_selection_changed(self) -> None:
        ext = self._selected()
        if ext is None:
            self.details.clear()
        else:
            state = repo.installed_state(ext, self.extensions_dir)
            header = (
                f"**Status:** {_STATE_TEXT[state]}"
                + (
                    "  \n*The installed file differs from the repository version.*"
                    if state == repo.STATE_UPDATE
                    else ""
                )
                + f"\n\n**Files:** {', '.join(ext.file_names)}\n\n---\n\n"
            )
            self.details.setMarkdown(header + (ext.readme or "*No readme provided.*"))
        self._update_buttons()

    def _update_buttons(self) -> None:
        ext = self._selected()
        busy = self._thread is not None and self._thread.isRunning()
        state = repo.installed_state(ext, self.extensions_dir) if ext else None

        self.install_button.setText(
            {repo.STATE_UPDATE: "Update", repo.STATE_INSTALLED: "Reinstall"}.get(
                state, "Install"
            )
        )
        self.install_button.setEnabled(ext is not None and not busy)
        self.uninstall_button.setEnabled(
            ext is not None and state in (repo.STATE_INSTALLED, repo.STATE_UPDATE)
        )
        self.web_button.setEnabled(self._spec is not None)

    # ----------------------------------------------------------------------- actions
    def _is_default_repo(self) -> bool:
        try:
            return repo.parse_repo_url(self.repo_edit.text()) == repo.parse_repo_url(
                repo.DEFAULT_REPO_URL
            )
        except repo.ExtensionRepoError:
            return False

    def _install_selected(self) -> None:
        ext = self._selected()
        if ext is None or self._spec is None:
            return
        state = repo.installed_state(ext, self.extensions_dir)
        verb = {repo.STATE_UPDATE: "Update", repo.STATE_INSTALLED: "Reinstall"}.get(
            state, "Install"
        )
        warning = (
            "Extensions are Python scripts that run inside VisoMaster with full access "
            "to your computer and files. Only install extensions from sources you trust."
        )
        if not self._is_default_repo():
            warning += (
                "\n\nThis is NOT the default repository. Make sure you trust its author."
            )
        overwrite = (
            "\n\nExisting files with the same name will be overwritten."
            if state != repo.STATE_NOT_INSTALLED
            else ""
        )
        answer = QtWidgets.QMessageBox.question(
            self,
            f"{verb} extension",
            f"{verb} \"{ext.name}\"?\n\nSource: {self._spec.label}\n"
            f"Files: {', '.join(ext.file_names)}\n\n{warning}{overwrite}",
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        try:
            repo.install_extension(ext, self.extensions_dir)
        except repo.ExtensionRepoError as e:
            QtWidgets.QMessageBox.warning(self, "Extension Manager", str(e))
            return
        done = {"Install": "Installed", "Update": "Updated", "Reinstall": "Reinstalled"}[verb]
        self._mark_changed(f"{done} \"{ext.name}\".")

    def _uninstall_selected(self) -> None:
        ext = self._selected()
        if ext is None:
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            "Uninstall extension",
            f"Remove \"{ext.name}\" ({', '.join(ext.file_names)}) from your extensions folder?",
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            repo.uninstall_extension(ext, self.extensions_dir)
        except repo.ExtensionRepoError as e:
            QtWidgets.QMessageBox.warning(self, "Extension Manager", str(e))
            return
        self._mark_changed(f"Removed \"{ext.name}\".")

    def _mark_changed(self, message: str) -> None:
        self._restart_needed = True
        self.restart_label.show()
        self.status_label.setText(message)
        self._populate_table()

    def _open_web(self) -> None:
        if self._spec is not None:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(self._spec.web_url))

    def _open_folder(self) -> None:
        self.extensions_dir.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.extensions_dir)))

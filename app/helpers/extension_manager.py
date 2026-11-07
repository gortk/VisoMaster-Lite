import sys
import traceback
import importlib.util
from pathlib import Path

class ExtensionManager:
    def __init__(self, main_window):
        self.main_window = main_window
        # Create an "extensions" folder in the project root
        self.extensions_dir = main_window.project_root_path / "extensions"
        if not self.extensions_dir.exists():
            self.extensions_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_extensions = []

    def load_all(self):
        """Scans the extensions folder and loads all .py files."""
        self.loaded_extensions.clear()
        for ext_file in self.extensions_dir.glob("*.py"):
            if ext_file.name.startswith("_"):
                continue
            self._load_extension(ext_file)

    def _load_extension(self, filepath: Path):
        try:
            module_name = f"viso_ext_{filepath.stem}"
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Extensions must define a `setup(main_window)` function
            if hasattr(module, "setup"):
                ext_instance = module.setup(self.main_window)
                if ext_instance:
                    self.loaded_extensions.append(ext_instance)
                print(f"[EXTENSION] Loaded: {filepath.name}")
            else:
                print(f"[EXTENSION] Skipped (no 'setup' function): {filepath.name}")
        except Exception as e:
            print(f"[EXTENSION] Error loading {filepath.name}: {e}")
            traceback.print_exc()

    def trigger_hook(self, hook_name: str, *args, **kwargs):
        """Calls a specific hook function on all loaded extensions."""
        for ext in self.loaded_extensions:
            if hasattr(ext, hook_name):
                try:
                    getattr(ext, hook_name)(*args, **kwargs)
                except Exception as e:
                    print(f"[EXTENSION] Error in {hook_name}: {e}")
                    traceback.print_exc()
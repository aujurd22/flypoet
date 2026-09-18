"""Patch adaptive_kwta.py: direct pyd loading before falling back to compile."""
import re

p = r"D:\user\flypoet\adaptive_kwta.py"
s = open(p, encoding="utf-8").read()

old = '''def _get_mod():
    """Lazy compile: importing this module must not require MSVC/nvcc."""
    global mod
    if mod is None:
        mod = load_inline('''
new = '''def _get_mod():
    """Lazy load: try prebuilt .pyd first (no MSVC needed), else compile."""
    global mod
    if mod is None:
        import importlib.util
        pyd = (r"C:\\Users\\user\\AppData\\Local\\torch_extensions"
               r"\\torch_extensions\\Cache\\py313_cu118\\adaptive_kwta_v1"
               r"\\adaptive_kwta_v1.pyd")
        if os.path.exists(pyd):
            spec = importlib.util.spec_from_file_location("adaptive_kwta_v1", pyd)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        mod = load_inline('''
assert old in s, "anchor not found"
s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
print("patched OK")

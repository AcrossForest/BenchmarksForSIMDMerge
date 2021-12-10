from pathlib import Path
builtDir = Path(__file__).parent.parent / "build"
projectRoot = Path(__file__).parent.parent
def exeName(s:str):
    return s
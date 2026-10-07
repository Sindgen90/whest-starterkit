"""Preserve the upstream MIT license and provenance in the official package."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parents[2]
archive = root/"submissions"/"phase2_best_20261001.tar.gz"
source = root/"research"/"phase2"/"venture_runtime"
with tarfile.open(archive,"r:gz") as original:
    files = {name:original.extractfile(name).read() for name in ("estimator.py","manifest.json")}
assert files["estimator.py"] == (source/"estimator.py").read_bytes()
for name in ("LICENSE","PROVENANCE.md"):
    files[name] = (source/name).read_bytes()
manifest = json.loads(files["manifest.json"])
manifest["files"] = [{"name":name,"sha256":hashlib.sha256(value).hexdigest()}
                     for name,value in files.items() if name!="manifest.json"]
files["manifest.json"] = json.dumps(manifest,indent=2).encode("utf-8")
temporary = archive.with_name(archive.name+".tmp")
with tarfile.open(temporary,"w:gz") as target:
    for name,value in files.items():
        info = tarfile.TarInfo(name)
        info.size = len(value)
        info.mode = 0o644
        target.addfile(info,io.BytesIO(value))
temporary.replace(archive)
with tarfile.open(archive,"r:gz") as packaged:
    for item in manifest["files"]:
        assert hashlib.sha256(packaged.extractfile(item["name"]).read()).hexdigest()==item["sha256"]
print(json.dumps({"archive":str(archive),"sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),"files":list(files)},indent=2))

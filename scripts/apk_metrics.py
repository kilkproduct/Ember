#!/usr/bin/env python3
"""Read actual produced APK sizes; no estimates or compression assumptions."""
import json,sys,zipfile
from pathlib import Path
rows=[]
for path in sys.argv[1:]:
 p=Path(path)
 with zipfile.ZipFile(p) as z:
  so={i.filename:{'bytes':i.file_size,'compressed':i.compress_size} for i in z.infolist() if i.filename.endswith('.so')}
  rows.append({'apk':p.name,'bytes':p.stat().st_size,'nativeBytes':sum(v['bytes'] for v in so.values()),'libraries':so,
   'pythonAndAssetsBytes':sum(i.file_size for i in z.infolist() if i.filename.startswith('assets/'))})
print(json.dumps(rows,indent=2))

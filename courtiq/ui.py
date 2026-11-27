"""Assemble the reviewed interface from editable source and attributed assets."""
import base64
from functools import lru_cache
import json
from pathlib import Path
import re


PACKAGE = Path(__file__).resolve().parent


def _embed_json(document, element_id, payload):
    encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    pattern = r'(<script id="' + re.escape(element_id) + r'" type="application/json">)[\s\S]*?(</script>)'
    result, count = re.subn(pattern, lambda match: match.group(1) + encoded + match.group(2),
                           document, count=1)
    if count != 1:
        raise ValueError('Missing interface data slot: ' + element_id)
    return result


@lru_cache(maxsize=1)
def load_html():
    """Return one self-contained page with the same photos, credits and facts.

    Raw photo files stay separate in the repository. Embedding them at startup
    keeps the existing image rendering and attribution UI intact without a
    separate public asset route or an external image request.
    """
    document = (PACKAGE / 'static/index.html').read_text(encoding='utf-8')
    photo_directory = PACKAGE / 'assets/players'
    manifest = json.loads((photo_directory / 'attribution.json').read_text(encoding='utf-8'))
    portraits = {}
    for player, record in manifest.items():
        filename = record['file']
        if Path(filename).name != filename or not filename.endswith('.jpg'):
            raise ValueError('Invalid portrait asset filename.')
        image = base64.b64encode((photo_directory / filename).read_bytes()).decode('ascii')
        portraits[player] = {
            ('data_uri' if key == 'file' else key):
            ('data:image/jpeg;base64,' + image if key == 'file' else value)
            for key, value in record.items()
        }
    facts = json.loads((PACKAGE / 'assets/tennis_facts.json').read_text(encoding='utf-8'))
    document = _embed_json(document, 'portrait-data', portraits)
    return _embed_json(document, 'tennis-facts-data', facts)

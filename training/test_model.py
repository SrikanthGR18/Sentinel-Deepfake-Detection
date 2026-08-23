import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.video_model import get_model

model = get_model()

print(model)
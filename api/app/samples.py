from typing import Dict, List

from PIL import Image, ImageDraw

from .schemas import SampleMetadata

SAMPLE_SIZE = (256, 256)

_SAMPLES: Dict[str, SampleMetadata] = {
    "coastal-flood-demo": SampleMetadata(
        id="coastal-flood-demo",
        title="Synthetic coastal flood",
        description="Toy RGB scene with bright flood-water regions near a coastline.",
        width=SAMPLE_SIZE[0],
        height=SAMPLE_SIZE[1],
        tags=["flood", "coast", "demo"],
    ),
    "wildfire-smoke-demo": SampleMetadata(
        id="wildfire-smoke-demo",
        title="Synthetic wildfire smoke",
        description="Toy RGB scene with pale smoke plumes and a burn scar.",
        width=SAMPLE_SIZE[0],
        height=SAMPLE_SIZE[1],
        tags=["wildfire", "smoke", "demo"],
    ),
    "urban-heat-demo": SampleMetadata(
        id="urban-heat-demo",
        title="Synthetic urban heat",
        description="Toy RGB scene with bright urban blocks surrounded by vegetation.",
        width=SAMPLE_SIZE[0],
        height=SAMPLE_SIZE[1],
        tags=["urban", "heat", "demo"],
    ),
}


def list_samples() -> List[SampleMetadata]:
    return list(_SAMPLES.values())


def get_sample(sample_id: str) -> SampleMetadata:
    return _SAMPLES[sample_id]


def has_sample(sample_id: str) -> bool:
    return sample_id in _SAMPLES


def build_sample_image(sample_id: str) -> Image.Image:
    if sample_id == "coastal-flood-demo":
        return _coastal_flood_image()
    if sample_id == "wildfire-smoke-demo":
        return _wildfire_smoke_image()
    if sample_id == "urban-heat-demo":
        return _urban_heat_image()
    raise KeyError(sample_id)


def _coastal_flood_image() -> Image.Image:
    image = Image.new("RGB", SAMPLE_SIZE, (42, 96, 64))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 156, 255, 255), fill=(43, 82, 150))
    draw.polygon([(0, 144), (58, 128), (116, 138), (190, 112), (255, 126), (255, 160), (0, 172)], fill=(186, 202, 214))
    for x in range(18, 240, 42):
        draw.rectangle((x, 36, x + 24, 76), fill=(102, 112, 96))
    draw.ellipse((82, 84, 184, 178), fill=(205, 218, 225))
    draw.ellipse((154, 130, 232, 206), fill=(196, 212, 224))
    return image


def _wildfire_smoke_image() -> Image.Image:
    image = Image.new("RGB", SAMPLE_SIZE, (80, 92, 55))
    draw = ImageDraw.Draw(image)
    draw.polygon([(14, 176), (120, 118), (240, 168), (226, 242), (42, 236)], fill=(78, 55, 43))
    draw.ellipse((58, 58, 182, 132), fill=(220, 220, 210))
    draw.ellipse((106, 30, 236, 106), fill=(231, 230, 218))
    draw.ellipse((22, 88, 148, 158), fill=(211, 212, 205))
    draw.rectangle((0, 210, 255, 255), fill=(66, 75, 48))
    return image


def _urban_heat_image() -> Image.Image:
    image = Image.new("RGB", SAMPLE_SIZE, (44, 104, 62))
    draw = ImageDraw.Draw(image)
    for x in range(18, 228, 42):
        for y in range(22, 212, 38):
            color = (178, 174, 160) if (x + y) % 3 else (214, 204, 176)
            draw.rectangle((x, y, x + 26, y + 22), fill=color)
    draw.line((0, 132, 255, 112), fill=(96, 96, 90), width=10)
    draw.line((128, 0, 146, 255), fill=(102, 102, 96), width=9)
    draw.ellipse((168, 150, 236, 218), fill=(224, 212, 176))
    return image

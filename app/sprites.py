"""Expression sprite sheets: one character, many expressions, in a grid.

Default layout, always in this order: stressed, accomplished, chill, confused
(1024 px cells, abstract background matching the character's vibe). Office
expressions and props are opt-in presets; up to 4 `activities` from the request
are appended as activity_1..4.

Consistency comes from image-to-image: every expression starts from the same
base portrait (generated, or supplied by the caller) with the same seed, so the
character, outfit, palette and framing carry over.

Expressive faces with hand gestures need a lot of change per cell, so the
default strength is 0.78. That only keeps identity because of three things
(tuned on Z-Image Turbo + the pixel LoRA):
  * a close-up chest-up framing with the hands near the face, so a gesture is a
    small move from the base pose (from a bust with no hands, or hands at the
    frame edge, gestures don't appear below ~0.75 and identity breaks at 0.8);
  * a neutral base with the hands clasped in front of the chest;
  * a concrete character description, outfit included ("plain dark green
    hooded cloak with no trim"); vague outfits drift (stray trim, colours).
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageEnhance
from pydantic import BaseModel, Field, field_validator, model_validator

PIXEL_LORA = "tarn59/pixel_art_style_lora_z_image_turbo"
DEFAULT_TEMPLATE = (
    "Pixel art style. Close-up chest-up portrait of {character}, face large in frame, "
    "{expression}, hands visible near the face, exaggerated expressive anime reaction, "
    "16-bit game character portrait, vibrant saturated bold colors, high contrast, "
    "simple abstract background whose colors and shapes fit the character's mood in this moment, "
    "no room, no scenery"
)
# Without "no room, no scenery" the model drew a wood-panelled office. Cells
# inherit the base portrait's backdrop, so it fits the character, not each cell.

# The base portrait every cell is derived from. Not a cell itself.
BASE_POSE = "calm neutral expression, hands loosely clasped together in front of the chest"

# The default sheet: four work moods, each a face plus a gesture or prop.
EXPRESSION_PRESETS: dict[str, str] = {
    "stressed": "stressed and frazzled, working frantically, wide panicked eyes, sweat drops flying, "
    "messy hair, typing furiously on a laptop",
    "accomplished": "finally done, triumphant relieved grin, eyes shut with satisfaction, pumping a fist in victory",
    "chill": "relaxed and chill, calm content half-smile, waiting for the next task, lazily sipping a mug of coffee",
    "confused": "confused, asking a question, one hand raised as if asking, one eyebrow raised, head tilted, "
    "question mark floating above",
}

# Opt-in extras (name them in `expressions`): office worker talking to the boss.
OFFICE_PRESETS: dict[str, str] = {
    "listening": "attentively listening, polite slight smile, nodding, one hand raised near the chin",
    "confident": "confident grin, bright eyes, explaining an idea, one hand raised open palm up",
    "nervous": "nervous, sweat drops on the forehead, awkward forced smile, tugging at the shirt collar",
    # Hands rarely show for this one; the teary, guilty face carries it.
    "apologetic": "apologizing, teary eyes, eyebrows slanted up, wobbly guilty smile, sweat drop, head bowed low",
}

# Props and activities with tuned wording.
PROP_PRESETS: dict[str, str] = {
    "reading": "focused, reading a printed report held up in both hands",
    "coffee": "relaxed, sipping from a white coffee mug held in both hands, steam rising",
    "coding": "concentrated, typing on an open laptop in front of the chest, screen glow on the face",
    "phone": "talking on a smartphone held to the ear, other hand gesturing",
    "notes": "taking notes, writing with a pen on a small notepad",
    "files": "struggling to carry a tall stack of paper folders held in both arms against the chest",
    "lunch": "eating lunch, taking a big bite of a sandwich held in both hands",
    "presenting": "presenting, holding up a small bar chart sign and pointing at it",
}

PRESETS: dict[str, str] = {**EXPRESSION_PRESETS, **OFFICE_PRESETS, **PROP_PRESETS}
DEFAULT_EXPRESSIONS = list(EXPRESSION_PRESETS)  # fixed order: stressed, accomplished, chill, confused
MAX_ACTIVITIES = 4
# Appended to caller-supplied activities: bare text ("drinking from a water
# bottle") often drew no prop at all; asking for it explicitly fixes that.
ACTIVITY_SUFFIX = "with the prop clearly visible and held in the hands near the face"


class ExpressionSpec(BaseModel):
    name: str = Field(..., min_length=1, max_length=40)
    prompt: str | None = Field(None, max_length=500, description="Defaults to the preset wording for `name`")
    strength: float | None = Field(None, gt=0, le=1)


class SpriteSheetRequest(BaseModel):
    character: str = Field(..., min_length=1, max_length=1000)
    expressions: list[str | ExpressionSpec] = Field(
        default_factory=lambda: list(DEFAULT_EXPRESSIONS), min_length=0, max_length=16
    )
    activities: list[str] = Field(
        default_factory=list,
        json_schema_extra={"items": {"maxLength": 300}},
        max_length=MAX_ACTIVITIES,
        description="Up to 4 caller-chosen props/activities, appended after `expressions`",
    )
    image: str | None = Field(None, description="Ready-made base portrait (base64 PNG), used as-is")
    sprite: str | None = Field(
        None, description="Reference sprite (base64 PNG, any size, transparency ok); styles the base portrait"
    )
    sprite_strength: float = Field(0.8, gt=0, le=1)
    vibrance: float = Field(1.35, ge=0.5, le=2.5)
    pixelate: bool = False
    pixel_grid: int = Field(128, ge=32, le=256)
    palette_colors: int = Field(32, ge=2, le=256)
    prompt_template: str = Field(DEFAULT_TEMPLATE, max_length=2000)
    strength: float = Field(0.78, gt=0, le=1)
    cell_size: int = Field(1024, ge=256, le=1024)
    render_size: int = Field(768, ge=256, le=1024)
    columns: int | None = Field(None, ge=1, le=8)
    gap: int = Field(0, ge=0, le=64)
    background: str = Field("#000000", pattern="^#[0-9A-Fa-f]{6}$")
    labels: bool = False
    seed: int | None = Field(None, ge=0, le=2**32 - 1)
    steps: int | None = Field(None, ge=1, le=50)
    # "auto": the pixel LoRA for text-only sheets; none with a `sprite`, where
    # the LoRA's anime look overrides the sprite's own (tested).
    lora: str | None = "auto"
    lora_strength: float = Field(1.0, ge=0, le=2)
    model: str | None = None
    format: str | None = Field(None, pattern="^(png|json)$")

    @field_validator("cell_size", "render_size")
    @classmethod
    def multiple_of_16(cls, v: int) -> int:
        if v % 16:
            raise ValueError("must be a multiple of 16")
        return v

    @field_validator("prompt_template")
    @classmethod
    def has_placeholders(cls, v: str) -> str:
        if "{character}" not in v or "{expression}" not in v:
            raise ValueError("must contain {character} and {expression}")
        return v

    @model_validator(mode="after")
    def unique_names(self) -> "SpriteSheetRequest":
        names = [e.name for e in self.resolved_expressions()]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"expression names must be unique; repeated: {duplicates}")
        if len(names) > 16:
            raise ValueError(f"at most 16 cells (expressions + activities); got {len(names)}")
        return self

    def resolved_expressions(self) -> list[ExpressionSpec]:
        """Cells in order (expressions, then activities), with stable names: a
        preset is always named by its lowercase key, so `name` identifies the same
        cell content in every response. Free text keeps its trimmed text as name."""
        out = []
        for item in self.expressions:
            spec = ExpressionSpec(name=item) if isinstance(item, str) else item
            raw = spec.name.strip()
            key = raw.lower()
            name = key if key in PRESETS else raw
            # Unknown names are used as free text ("pouting, puffed cheeks").
            prompt = spec.prompt or PRESETS.get(key, raw)
            out.append(ExpressionSpec(name=name, prompt=prompt, strength=spec.strength or self.strength))
        # Activity cells get fixed slot names (activity_1..4) so clients can map
        # cells without depending on the text they sent; `prompt` carries the text.
        for i, activity in enumerate(self.activities, start=1):
            text = activity.strip()
            if not text:
                raise ValueError("activities must not be empty strings")
            out.append(ExpressionSpec(name=f"activity_{i}", prompt=f"{text}, {ACTIVITY_SUFFIX}", strength=self.strength))
        return out

    @model_validator(mode="after")
    def one_base_source(self) -> "SpriteSheetRequest":
        if self.image is not None and self.sprite is not None:
            raise ValueError("send either `image` (a ready base portrait) or `sprite`, not both")
        return self

    def effective_lora(self) -> str | None:
        if self.lora == "auto":
            return None if self.sprite is not None else PIXEL_LORA
        return self.lora

    def base_prompt(self) -> str:
        return self.prompt_template.replace("{character}", self.character).replace("{expression}", BASE_POSE)

    def prompt_for(self, expression: ExpressionSpec) -> str:
        return self.prompt_template.replace("{character}", self.character).replace(
            "{expression}", expression.prompt
        )


def prep_sprite(sprite: Image.Image, size: int, background: tuple[int, int, int] = (20, 20, 28)) -> Image.Image:
    """Turn a small game sprite into a chest-up reference: flatten transparency
    onto a dark background, crop the head and chest (top 60% of the visible
    figure), and scale up with hard pixel edges."""
    sprite = sprite.convert("RGBA")
    bbox = sprite.getbbox() or (0, 0, *sprite.size)
    x0, y0, x1, y1 = bbox
    side = max(x1 - x0, int((y1 - y0) * 0.6), 1)
    cx = (x0 + x1) // 2
    crop = sprite.crop((cx - side // 2, y0, cx - side // 2 + side, y0 + side))
    flat = Image.new("RGB", crop.size, background)
    flat.paste(crop, mask=crop.split()[3])
    return flat.resize((size, size), Image.NEAREST)


def boost_colors(cells: list[Image.Image], vibrance: float) -> list[Image.Image]:
    """Saturation x `vibrance`, plus a third of that as contrast. Runs before
    pixelate, whose shared palette otherwise averages colours toward dull mid-tones."""
    if vibrance == 1.0:
        return cells
    contrast = 1 + (vibrance - 1) / 3
    return [ImageEnhance.Contrast(ImageEnhance.Color(c).enhance(vibrance)).enhance(contrast) for c in cells]


def pixelate(cells: list[Image.Image], grid: int, colors: int, size: int) -> list[Image.Image]:
    """Snap cells to a real pixel grid with one palette shared by the whole
    sheet, so the same colour stays the same colour in every cell."""
    small = [c.resize((grid, grid), Image.BOX) for c in cells]
    mosaic = Image.new("RGB", (grid * len(small), grid))
    for i, c in enumerate(small):
        mosaic.paste(c, (i * grid, 0))
    palette = mosaic.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    return [
        c.quantize(palette=palette, dither=Image.Dither.NONE).convert("RGB").resize((size, size), Image.NEAREST)
        for c in small
    ]


def compose_sheet(
    cells: list[Image.Image],
    names: list[str],
    *,
    columns: int,
    cell: int,
    gap: int,
    background: str,
    labels: bool,
) -> tuple[Image.Image, list[dict]]:
    rows = -(-len(cells) // columns)
    label_h = max(14, cell // 16) if labels else 0
    pitch_y = cell + label_h
    sheet = Image.new(
        "RGB",
        (columns * cell + (columns + 1) * gap, rows * pitch_y + (rows + 1) * gap),
        background,
    )
    draw = ImageDraw.Draw(sheet)
    boxes = []
    for i, (image, name) in enumerate(zip(cells, names)):
        r, c = divmod(i, columns)
        x, y = gap + c * (cell + gap), gap + r * (pitch_y + gap)
        sheet.paste(image.resize((cell, cell), Image.NEAREST), (x, y))
        if labels:
            draw.text((x + 4, y + cell + 1), name, fill=_contrast(background))
        boxes.append({"name": name, "x": x, "y": y, "width": cell, "height": cell})
    return sheet, boxes


def _contrast(hex_color: str) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return "#000000" if (r * 299 + g * 587 + b * 114) / 1000 > 128 else "#ffffff"

"""Photographic styles: a whole look chosen in one tap.

A style is a template plus a set of preset option values.  The templates take
{identity}, {changes}, {preserve}, {scene}, {lighting} and {camera}, which
generation/prompt.py fills; nothing here writes a finished prompt, so the
identity clause can never be accidentally dropped by a style.
"""
from __future__ import annotations

# Templates are plain strings on purpose: generation/prompt.py fills the
# placeholders, so a style can never accidentally drop the identity clause.
_BASE_NEG = ("blurry, low resolution, oversaturated, harsh digital sharpening, "
             "watermark, text overlay, distorted background, warped lines")


def _s(key: str, es: str, en: str, desc: str, shots: str, prompt: str,
       negative: str = "", defaults: dict | None = None,
       params: dict | None = None, sort: int = 0) -> dict:
    return {
        "key": key, "name_es": es, "name_en": en, "description": desc,
        "shot_types": shots,
        "prompt_template": prompt,
        "negative_template": ", ".join(p for p in (negative, _BASE_NEG) if p),
        "defaults": defaults or {},
        "params": params or {"strength": 0.5, "guidance": 4.0, "steps": 28},
        "sort_order": sort,
    }


BUILTIN_STYLES: list[dict] = [
    _s("editorial_moda", "Editorial de moda", "Fashion editorial",
       "Como una revista de moda: luz cuidada, pose intencionada, fondo limpio.",
       "half,full",
       "{identity}, {changes}, {preserve}, a high end fashion editorial "
       "photograph, {scene}, {lighting}, {camera}, styled and deliberate, "
       "magazine quality",
       "amateur snapshot, cluttered background",
       {"grade": ["editorial_frio"], "lighting": ["ventana_izq"]},
       {"strength": 0.52, "guidance": 4.2, "steps": 30}, 10),

    _s("retrato_corporativo", "Retrato corporativo", "Corporate portrait",
       "Para LinkedIn y perfiles profesionales: cercano, serio y bien iluminado.",
       "closeup,half",
       "{identity}, {changes}, {preserve}, a professional corporate "
       "headshot, {scene}, {lighting}, {camera}, approachable and competent",
       "party background, heavy makeup, dramatic shadows",
       {"lighting": ["softbox"], "grade": ["estudio_neutro"],
        "expression": ["sonrisa_suave"]},
       {"strength": 0.42, "guidance": 3.8, "steps": 28}, 20),

    _s("retrato_estudio", "Retrato de estudio", "Studio portrait",
       "Fondo liso y luz controlada. El clasico que siempre funciona.",
       "closeup,half,full",
       "{identity}, {changes}, {preserve}, a classic studio portrait, "
       "{scene}, {lighting}, {camera}, clean and timeless",
       "busy background",
       {"scene": ["estudio_gris"], "lighting": ["softbox"]},
       {"strength": 0.45, "guidance": 4.0, "steps": 28}, 30),

    _s("belleza_natural", "Belleza natural", "Natural beauty",
       "Piel real, luz suave y ningun retoque. Tu cara tal cual es.",
       "closeup",
       "{identity}, {changes}, {preserve}, a natural beauty portrait, "
       "{scene}, {lighting}, {camera}, real skin texture with visible pores and "
       "fine lines, no retouching",
       "airbrushed skin, plastic skin, heavy retouching, beauty filter",
       {"lighting": ["difusa"], "grade": ["estudio_neutro"]},
       {"strength": 0.35, "guidance": 3.5, "steps": 30}, 40),

    _s("lookbook_urbano", "Lookbook urbano", "Urban lookbook",
       "En la calle, con la ropa como protagonista.", "full",
       "{identity}, {changes}, {preserve}, an urban street style lookbook "
       "photograph, {scene}, {lighting}, {camera}, candid and current",
       "studio backdrop",
       {"scene": ["hormigon"], "pose": ["caminando"]},
       {"strength": 0.55, "guidance": 4.2, "steps": 30}, 50),

    _s("campana_producto", "Campana de producto", "Product campaign",
       "Para mostrar un objeto o una prenda concreta con la persona.",
       "half,full",
       "{identity}, {changes}, {preserve}, a polished product campaign "
       "photograph, {scene}, {lighting}, {camera}, the product clearly readable",
       "cluttered composition",
       {"scene": ["ciclorama_blanco"], "lighting": ["softbox"]},
       {"strength": 0.5, "guidance": 4.0, "steps": 30}, 60),

    _s("editorial_bn", "Editorial en blanco y negro", "Black and white editorial",
       "Sin color: solo luz, forma y expresion.", "closeup,half,full",
       "{identity}, {changes}, {preserve}, a black and white editorial "
       "photograph, {scene}, {lighting}, {camera}, rich tonal range, fine grain",
       "colour cast",
       {"grade": ["blanco_negro"]},
       {"strength": 0.5, "guidance": 4.0, "steps": 30}, 70),

    _s("cine_noir", "Cine noir", "Film noir",
       "Sombras marcadas y mucho contraste, como el cine antiguo.",
       "closeup,half",
       "{identity}, {changes}, {preserve}, a film noir portrait, {scene}, "
       "{lighting}, {camera}, deep shadows and hard key light",
       "flat lighting, bright cheerful tone",
       {"lighting": ["luz_dura"], "grade": ["blanco_negro"],
        "scene": ["fondo_oscuro"]},
       {"strength": 0.55, "guidance": 4.5, "steps": 32}, 80),

    _s("hora_dorada", "Hora dorada", "Golden hour",
       "La luz calida del atardecer, favorecedora y natural.",
       "half,full",
       "{identity}, {changes}, {preserve}, a golden hour portrait outdoors, "
       "{scene}, {lighting}, {camera}, warm rim light and long shadows",
       "midday harsh light, cold tone",
       {"lighting": ["hora_dorada"], "scene": ["cielo_dorado"],
        "grade": ["film_calido"]},
       {"strength": 0.5, "guidance": 4.0, "steps": 30}, 90),

    _s("alfombra_roja", "Alfombra roja", "Red carpet",
       "De gala, con vestido largo y presencia.", "full",
       "{identity}, {changes}, {preserve}, a red carpet arrival "
       "photograph, {scene}, {lighting}, {camera}, glamorous and poised",
       "casual clothing, daylight",
       {"clothing": ["vestido_noche"], "lighting": ["luz_dura"]},
       {"strength": 0.55, "guidance": 4.2, "steps": 32}, 100),

    _s("minimal_escandinavo", "Minimalista escandinavo", "Scandinavian minimal",
       "Poca cosa, mucha luz y colores suaves.", "half,full",
       "{identity}, {changes}, {preserve}, a minimal scandinavian style "
       "photograph, {scene}, {lighting}, {camera}, restrained palette, calm",
       "saturated colours, busy set",
       {"scene": ["luz_ventana"], "grade": ["pastel"]},
       {"strength": 0.48, "guidance": 3.8, "steps": 28}, 110),

    _s("vintage_analogico", "Vintage analogico", "Analogue vintage",
       "Con el grano y el color de la fotografia de carrete.",
       "closeup,half,full",
       "{identity}, {changes}, {preserve}, an analogue film photograph, "
       "{scene}, {lighting}, {camera}, visible grain, slightly faded colour",
       "digital clean look, hdr",
       {"grade": ["film_calido"]},
       {"strength": 0.5, "guidance": 4.0, "steps": 28}, 120),

    _s("alta_costura", "Alta costura", "Haute couture",
       "Muy editorial, casi escultural. Para fotos con mucha fuerza.",
       "full",
       "{identity}, {changes}, {preserve}, a haute couture editorial "
       "photograph, {scene}, {lighting}, {camera}, sculptural silhouette, "
       "dramatic staging",
       "casual clothing, snapshot feel",
       {"lighting": ["luz_dura"], "scene": ["fondo_oscuro"]},
       {"strength": 0.58, "guidance": 4.5, "steps": 34}, 130),

    _s("streetwear", "Streetwear", "Streetwear",
       "Ropa urbana, actitud relajada, calle de fondo.", "half,full",
       "{identity}, {changes}, {preserve}, a streetwear photograph, "
       "{scene}, {lighting}, {camera}, relaxed contemporary attitude",
       "formalwear, studio backdrop",
       {"clothing": ["vaqueros_camiseta"], "scene": ["ladrillo"]},
       {"strength": 0.55, "guidance": 4.2, "steps": 30}, 140),

    _s("retrato_ambiental", "Retrato ambiental", "Environmental portrait",
       "En un sitio real que cuenta algo de ti.", "half,full",
       "{identity}, {changes}, {preserve}, an environmental portrait in a "
       "real location, {scene}, {lighting}, {camera}, the setting tells a story",
       "plain studio backdrop",
       {"scene": ["luz_ventana"], "lighting": ["ventana_izq"]},
       {"strength": 0.5, "guidance": 4.0, "steps": 30}, 150),

    _s("ecommerce", "Catalogo e-commerce", "E-commerce catalogue",
       "Fondo blanco y ropa perfectamente visible, para vender.",
       "full",
       "{identity}, {changes}, {preserve}, a clean e-commerce catalogue "
       "photograph, {scene}, {lighting}, {camera}, garment fully visible, even "
       "lighting, no distractions",
       "dramatic shadows, coloured background, artistic crop",
       {"scene": ["ciclorama_blanco"], "lighting": ["softbox"],
        "pose": ["de_pie_frontal"]},
       {"strength": 0.45, "guidance": 3.8, "steps": 30}, 160),

    _s("foto_perfil", "Foto de perfil profesional", "Professional profile photo",
       "La foto de perfil que sirve para todo.", "closeup",
       "{identity}, {changes}, {preserve}, a friendly professional profile "
       "photograph, {scene}, {lighting}, {camera}, clear face, simple background",
       "sunglasses, heavy shadow across the face",
       {"lighting": ["softbox"], "expression": ["sonrisa_suave"],
        "scene": ["estudio_gris"]},
       {"strength": 0.4, "guidance": 3.8, "steps": 28}, 170),

    _s("revista_lifestyle", "Revista lifestyle", "Lifestyle magazine",
       "Natural y cotidiano, como un reportaje de revista.", "half,full",
       "{identity}, {changes}, {preserve}, a lifestyle magazine "
       "photograph, {scene}, {lighting}, {camera}, natural and unforced",
       "stiff posing, studio strobes",
       {"scene": ["luz_ventana"], "expression": ["risa_natural"]},
       {"strength": 0.5, "guidance": 4.0, "steps": 30}, 180),

    _s("glamour_clasico", "Glamour clasico", "Classic glamour",
       "Luz suave y favorecedora, elegante y atemporal.", "closeup,half",
       "{identity}, {changes}, {preserve}, a classic glamour portrait, "
       "{scene}, {lighting}, {camera}, soft flattering light, elegant",
       "harsh light, casual snapshot",
       {"lighting": ["difusa"], "hair": ["volumen_glam"]},
       {"strength": 0.48, "guidance": 4.0, "steps": 30}, 190),

    _s("retrato_dramatico", "Retrato dramatico", "Dramatic portrait",
       "Una sola luz dura y fondo oscuro. Mucha personalidad.",
       "closeup,half",
       "{identity}, {changes}, {preserve}, a dramatic low key portrait, "
       "{scene}, {lighting}, {camera}, single hard light, deep falloff",
       "flat even lighting",
       {"lighting": ["luz_dura"], "scene": ["fondo_oscuro"]},
       {"strength": 0.55, "guidance": 4.4, "steps": 32}, 200),

    _s("belleza_limpia", "Campana de belleza", "Clean beauty campaign",
       "Primerisimo plano de piel y mirada, sin trucos.", "closeup",
       "{identity}, {changes}, {preserve}, a clean beauty campaign close "
       "up, {scene}, {lighting}, {camera}, luminous real skin, precise focus on "
       "the eyes",
       "airbrushed skin, plastic texture, altered features",
       {"lighting": ["difusa"], "grade": ["estudio_neutro"]},
       {"strength": 0.35, "guidance": 3.6, "steps": 32}, 210),

    _s("boda_elegante", "Boda elegante", "Elegant wedding",
       "Luz calida y ambiente de celebracion.", "half,full",
       "{identity}, {changes}, {preserve}, an elegant wedding photograph, "
       "{scene}, {lighting}, {camera}, warm celebratory atmosphere",
       "harsh flash, cluttered background",
       {"lighting": ["hora_dorada"], "grade": ["film_calido"]},
       {"strength": 0.5, "guidance": 4.0, "steps": 30}, 220),
]

STYLES_BY_KEY = {s["key"]: s for s in BUILTIN_STYLES}


def styles_for_shot(shot_type: str) -> list[dict]:
    if shot_type not in ("closeup", "half", "full"):
        return list(BUILTIN_STYLES)
    return [s for s in BUILTIN_STYLES
            if shot_type in s["shot_types"].split(",")]


def get_style(key: str) -> dict | None:
    return STYLES_BY_KEY.get(str(key or ""))


def default_style(shot_type: str) -> dict:
    options = styles_for_shot(shot_type)
    return options[0] if options else BUILTIN_STYLES[0]

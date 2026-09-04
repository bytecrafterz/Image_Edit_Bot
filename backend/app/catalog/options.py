"""The menu of changes the user can ask for.

Labels are Spanish because she reads them; prompt fragments are English because
the models respond better to it.  Every fragment is a photographic phrase rather
than a single word - "a floor length evening gown in matte satin" produces a
different image from "dress", and the whole point of the robot is that she never
has to know that.

Each value also carries a ``local`` hint so the free local engine knows which
scene, grade or lighting rig to build for it, and every garment carries a
``garment`` classification saying whether it dresses the whole person or only
the top half - see the block below, which exists because a shirt that did not
say what the legs were wearing cost the client a paid image.
"""
from __future__ import annotations

import re
from typing import Any


# --------------------------------------------------- what a garment covers
#
# Eight of the eighteen garments below name only an upper body piece, and until
# 2026-09-04 nothing in the system said what the lower half was wearing.  The
# paid image bought as this project's delivery is what that costs.  Attempt
# att_d6fb1c97f9874a82b38d35c3 sent "change only: a crisp white cotton poplin
# shirt, sleeves lightly rolled" over a photograph of her in lingerie, and its
# stored negative prompt - 62 terms about slimming, fingers, seams and
# watermarks - does not contain the words underwear, lingerie, legs or
# trousers.  Nothing in that request ever asked for the lower half, so the
# engine kept the one that was already in the photograph: the shirt was painted
# on over the lingerie, bare legs, a dark lace edge showing at the hem.  The
# same option on the same photograph had produced black trousers on an earlier
# square preview, so the fragment was never wrong - it was silent, and a silent
# request is a coin toss.
#
# ``garment`` is the answer.  ``kind`` says whether the piece dresses the whole
# person (``complete``), only the top half (``top``) or only the bottom half
# (``bottom``); ``lower`` says what the lower half ends up being, which is what
# decides whether bare legs are a defect or simply a midi dress; ``bottom`` is
# the English phrase naming that lower half when the fragment does not already
# name it.  It lives BESIDE the garment instead of inside its prompt fragment
# on purpose: a bottom the user picks herself has to be able to replace it, and
# a phrase welded into the fragment cannot be replaced by anything.  The
# automatic completion only ever fills a gap - see prompt.outfit_plan.
#
# It travels in ``params`` because that is the one field seed.py already copies
# into the options table verbatim (``local`` rides the same way), so the
# classification reaches the database, the API and the report with no schema
# change.


def _complete(lower: str, bottom: str = "") -> dict:
    """A garment that dresses the whole person on its own."""
    return {"kind": "complete", "lower": lower, "bottom": bottom}


def _top(bottom: str, lower: str = "trousers") -> dict:
    """An upper body piece.  ``bottom`` is what completes it by default."""
    return {"kind": "top", "lower": lower, "bottom": bottom}


def _bottom(lower: str = "trousers") -> dict:
    """A lower body piece the user picked on purpose."""
    return {"kind": "bottom", "lower": lower, "bottom": ""}


def _v(key: str, es: str, en: str, prompt: str, negative: str = "",
       shots: str = "closeup,half,full", local: dict | None = None,
       params: dict | None = None, garment: dict | None = None) -> dict:
    merged = dict(params or {})
    if garment:
        merged["garment"] = dict(garment)
    return {"value_key": key, "label_es": es, "label_en": en,
            "prompt_fragment": prompt, "negative_fragment": negative,
            "shot_types": shots, "local": local or {}, "params": merged}


BUILTIN_OPTIONS: list[dict] = [
    {
        # Available on a close up too.  The collar, the shoulders and the
        # neckline are all in frame on a head and shoulders shot, and a
        # generative provider dresses them perfectly well - a blazer and shirt
        # on a headshot was one of the first things this system produced.  The
        # floor length pieces below stay restricted to wider framings, where
        # there is actually a skirt to see.
        "group_key": "clothing", "label_es": "Ropa", "label_en": "Clothing",
        "multi": True, "sort_order": 10, "shot_types": "closeup,half,full",
        "values": [
            # --- complete outfits: they already dress the whole person ---
            _v("vestido_noche", "Vestido de noche", "Evening gown",
               "a floor length evening gown in matte satin, elegant drape, subtle sheen",
               shots="full", garment=_complete("dress")),
            # "two piece suit" does not say which two pieces, and a jacket over
            # nothing is a legal reading of it.  Naming the trousers costs a
            # clause and removes the reading.
            _v("traje_sastre", "Traje sastre", "Tailored suit",
               "a sharply tailored two piece suit in fine wool, structured shoulders",
               garment=_complete("trousers",
                                 "matching full length tailored trousers")),
            _v("vaqueros_camiseta", "Vaqueros y camiseta", "Jeans and tee",
               "straight leg blue jeans with a plain white cotton t-shirt",
               garment=_complete("trousers")),
            _v("mono_elegante", "Mono elegante", "Elegant jumpsuit",
               "a tailored wide leg jumpsuit in deep navy, cinched at the waist",
               shots="full", garment=_complete("jumpsuit")),
            _v("falda_midi", "Falda midi", "Midi skirt",
               "a pleated midi skirt with a tucked in silk blouse", shots="full",
               garment=_complete("skirt")),
            _v("vestido_verano", "Vestido de verano", "Summer dress",
               "a light cotton summer dress with a small floral print", shots="full",
               garment=_complete("dress")),
            # "matching technical sportswear" reads just as easily as a sports
            # bra with nothing below it, which is the failure this catalogue
            # already paid for once.
            _v("deportiva_elegante", "Ropa deportiva elegante", "Elevated sportswear",
               "matching technical sportswear in a muted tone, clean minimal styling",
               garment=_complete("trousers", "matching full length leggings")),
            _v("conjunto_lino", "Conjunto de lino", "Linen set",
               "a relaxed linen shirt and trouser set in natural ecru",
               garment=_complete("trousers")),
            _v("esmoquin", "Esmoquin", "Tuxedo",
               "a womens tuxedo with satin lapels, crisp white shirt", shots="full",
               garment=_complete("trousers", "matching tuxedo trousers")),
            _v("vestido_rojo", "Vestido rojo largo", "Long red dress",
               "a long crimson dress in flowing crepe, simple neckline", shots="full",
               garment=_complete("dress")),

            # --- upper body only: each one names the lower half it needs ---
            # This is the group the delivered image came from.  Every default
            # below reaches the ankle or the knee on purpose: "trousers" alone
            # still leaves the engine free to stop at the hip.
            _v("blazer_oversize", "Blazer oversize", "Oversized blazer",
               "an oversized wool blazer worn open over a simple top, relaxed tailoring",
               garment=_top("matching tailored trousers in the same weight of "
                            "cloth, full length to the ankle")),
            _v("camisa_blanca", "Camisa blanca", "White shirt",
               "a crisp white cotton poplin shirt, sleeves lightly rolled",
               garment=_top("dark charcoal tailored trousers, full length to "
                            "the ankle, the shirt tucked in at the waistband")),
            _v("top_punto", "Top de punto", "Knit top",
               "a fine gauge knit top in a warm neutral tone, soft texture",
               garment=_top("dark straight leg trousers, full length to the ankle")),
            _v("abrigo_lana", "Abrigo de lana", "Wool coat",
               "a long wool coat in camel, clean lines, worn open",
               garment=_top("a fine knit top and dark straight leg trousers "
                            "worn underneath the open coat, full length to "
                            "the ankle")),
            _v("gabardina", "Gabardina", "Trench coat",
               "a classic belted trench coat in beige gabardine",
               garment=_top("dark tailored trousers worn underneath, full "
                            "length to the ankle, the coat belted and fully "
                            "fastened")),
            _v("chaqueta_cuero", "Chaqueta de cuero", "Leather jacket",
               "a black leather biker jacket over a plain top, worn open",
               garment=_top("a plain top and dark straight leg jeans, full "
                            "length to the ankle")),
            _v("jersey_cachemira", "Jersey de cachemira", "Cashmere sweater",
               "a soft oversized cashmere sweater in oatmeal",
               garment=_top("dark straight leg trousers, full length to the ankle")),
            _v("blusa_seda", "Blusa de seda", "Silk blouse",
               "a fluid silk blouse in ivory, softly draped",
               garment=_top("a dark tailored midi skirt to below the knee, "
                            "the blouse tucked in at the waistband",
                            lower="skirt")),

            # --- lower body only: choosing one of these overrides the
            # automatic completion above, which is the whole reason they are
            # in the same menu and not a separate one.  Absent on a close up:
            # there is no lower half in frame to dress.
            _v("pantalon_sastre", "Pantalon de vestir", "Tailored trousers",
               "dark tailored trousers, full length to the ankle, clean break",
               shots="half,full", garment=_bottom()),
            _v("vaqueros_rectos", "Vaqueros rectos", "Straight leg jeans",
               "straight leg blue jeans, full length to the ankle",
               shots="half,full", garment=_bottom()),
            _v("pantalon_ancho", "Pantalon ancho", "Wide leg trousers",
               "high waisted wide leg trousers, full length to the floor",
               shots="half,full", garment=_bottom()),
            # Not "slim cigarette trousers": prompt.BODY_CHANGE_RE reads the
            # word slim as a request to narrow her body and refuses the whole
            # option, which left this garment with no prompt fragment at all.
            # The trousers are described by their cut instead.
            _v("pantalon_pitillo", "Pantalon pitillo", "Cigarette trousers",
               "tapered cigarette trousers cropped at the ankle",
               shots="half,full", garment=_bottom()),
            _v("falda_tubo", "Falda de tubo", "Pencil skirt",
               "a tailored pencil skirt to just below the knee",
               shots="half,full", garment=_bottom("skirt")),
            _v("falda_larga", "Falda larga", "Long skirt",
               "a long flowing skirt falling to the ankle",
               shots="half,full", garment=_bottom("skirt")),
        ],
    },
    {
        "group_key": "clothing_color", "label_es": "Color de la ropa",
        "label_en": "Clothing colour", "multi": True, "sort_order": 20,
        "shot_types": "closeup,half,full",
        "values": [
            _v("negro", "Negro", "Black", "in deep black",
               local={"hsv": [0, 0, 45]}),
            _v("blanco", "Blanco", "White", "in clean white",
               local={"hsv": [0, 0, 230]}),
            _v("crudo", "Crudo", "Ecru", "in warm ecru", local={"hsv": [30, 40, 205]}),
            _v("camel", "Camel", "Camel", "in camel tan", local={"hsv": [22, 120, 165]}),
            _v("marino", "Azul marino", "Navy", "in navy blue",
               local={"hsv": [110, 150, 80]}),
            _v("azul_cielo", "Azul cielo", "Sky blue", "in soft sky blue",
               local={"hsv": [100, 90, 200]}),
            _v("verde_oliva", "Verde oliva", "Olive", "in olive green",
               local={"hsv": [35, 110, 120]}),
            _v("verde_esmeralda", "Verde esmeralda", "Emerald", "in emerald green",
               local={"hsv": [75, 160, 120]}),
            _v("burdeos", "Burdeos", "Burgundy", "in deep burgundy",
               local={"hsv": [175, 150, 95]}),
            _v("rojo", "Rojo", "Red", "in true red", local={"hsv": [178, 190, 150]}),
            _v("rosa_palo", "Rosa palo", "Dusty pink", "in dusty rose",
               local={"hsv": [172, 60, 200]}),
            _v("gris", "Gris", "Grey", "in mid grey", local={"hsv": [0, 0, 130]}),
            _v("chocolate", "Chocolate", "Chocolate", "in chocolate brown",
               local={"hsv": [12, 120, 90]}),
            _v("mostaza", "Mostaza", "Mustard", "in mustard yellow",
               local={"hsv": [22, 170, 180]}),
        ],
    },
    {
        "group_key": "hair", "label_es": "Peinado", "label_en": "Hair",
        "multi": True, "sort_order": 30,
        "values": [
            _v("suelto_liso", "Suelto liso", "Straight down",
               "hair worn down and straight, natural movement"),
            _v("ondas_suaves", "Ondas suaves", "Soft waves",
               "hair in soft loose waves"),
            _v("recogido_pulido", "Recogido pulido", "Sleek updo",
               "hair in a sleek polished updo"),
            _v("coleta_alta", "Coleta alta", "High ponytail",
               "hair in a high ponytail, clean at the temples"),
            _v("mono_bajo", "Mono bajo", "Low bun",
               "hair in a low chignon at the nape"),
            _v("trenza_lateral", "Trenza lateral", "Side braid",
               "hair in a loose side braid"),
            _v("semirecogido", "Semirecogido", "Half up",
               "hair half pinned back, the rest falling loose"),
            _v("rizos_definidos", "Rizos definidos", "Defined curls",
               "hair in defined natural curls"),
            _v("raya_al_medio", "Raya al medio", "Centre parting",
               "hair straight with a clean centre parting"),
            _v("volumen_glam", "Volumen glam", "Glam volume",
               "hair with full glamorous volume and bounce"),
        ],
    },
    {
        "group_key": "expression", "label_es": "Expresion",
        "label_en": "Expression", "multi": True, "sort_order": 40,
        "values": [
            _v("sonrisa_suave", "Sonrisa suave", "Soft smile",
               "a soft closed lip smile, relaxed eyes"),
            _v("mirada_seria", "Mirada seria", "Serious",
               "a calm serious expression, direct gaze"),
            _v("sonrisa_amplia", "Sonrisa amplia", "Broad smile",
               "a warm open smile"),
            _v("mirada_frente", "Mirada al frente", "Straight to camera",
               "looking straight into the lens, composed"),
            _v("perfil_pensativo", "Perfil pensativo", "Thoughtful profile",
               "looking away from the camera, thoughtful"),
            _v("risa_natural", "Risa natural", "Natural laugh",
               "caught mid laugh, natural and unposed"),
            _v("mirada_baja", "Mirada baja", "Downward gaze",
               "eyes lowered, quiet expression"),
            _v("actitud_segura", "Actitud segura", "Confident",
               "a confident assured expression, chin level"),
        ],
    },
    {
        "group_key": "pose", "label_es": "Postura", "label_en": "Pose",
        "multi": True, "sort_order": 50, "shot_types": "half,full",
        "values": [
            _v("de_pie_frontal", "De pie de frente", "Standing front",
               "standing squarely facing the camera, arms relaxed"),
            _v("tres_cuartos", "Tres cuartos", "Three quarter",
               "standing at a three quarter angle to the camera"),
            _v("perfil", "De perfil", "Profile", "turned in profile to the camera"),
            _v("brazos_cruzados", "Brazos cruzados", "Arms crossed",
               "arms lightly crossed, weight on one hip"),
            _v("mano_cadera", "Mano en la cadera", "Hand on hip",
               "one hand resting on the hip"),
            _v("caminando", "Caminando", "Walking",
               "captured mid stride, natural walking motion", shots="full"),
            _v("apoyada", "Apoyada", "Leaning",
               "leaning against a wall, relaxed posture"),
            _v("sentada", "Sentada", "Seated",
               "seated with an upright relaxed posture"),
            _v("mirando_hombro", "Mirando por encima del hombro", "Over shoulder",
               "looking back over one shoulder"),
            _v("manos_bolsillos", "Manos en los bolsillos", "Hands in pockets",
               "hands in pockets, shoulders relaxed"),
        ],
    },
    {
        "group_key": "scene", "label_es": "Escenario", "label_en": "Scene",
        "multi": True, "sort_order": 60,
        "values": [
            _v("estudio_gris", "Estudio gris", "Grey studio",
               "a seamless mid grey studio backdrop", local={"scene": "studio_gray"}),
            _v("estudio_calido", "Estudio calido", "Warm studio",
               "a seamless warm grey studio backdrop",
               local={"scene": "studio_warm_gray"}),
            _v("estudio_azul", "Estudio azul", "Blue studio",
               "a seamless dusty blue studio backdrop",
               local={"scene": "studio_blue"}),
            _v("estudio_terracota", "Estudio terracota", "Terracotta studio",
               "a seamless terracotta studio backdrop",
               local={"scene": "studio_terracotta"}),
            _v("ciclorama_blanco", "Ciclorama blanco", "White cyclorama",
               "a bright white cyclorama, clean and even",
               local={"scene": "white_cyclorama"}),
            _v("degradado_melocoton", "Degradado melocoton", "Peach gradient",
               "a smooth peach gradient backdrop",
               local={"scene": "gradient_peach"}),
            _v("degradado_indigo", "Degradado indigo", "Indigo gradient",
               "a smooth indigo gradient backdrop",
               local={"scene": "gradient_indigo"}),
            _v("bokeh_calido", "Luces bokeh calidas", "Warm bokeh",
               "warm golden bokeh lights far behind the subject",
               local={"scene": "bokeh_warm", "blur_background": 0.85}),
            _v("ciudad_noche", "Ciudad de noche", "City at night",
               "a city street at night, distant lights thrown out of focus",
               local={"scene": "bokeh_city", "blur_background": 0.9}),
            _v("playa_atardecer", "Playa al atardecer", "Beach at sunset",
               "a wide beach at golden hour, soft haze on the horizon",
               local={"scene": "beach_haze"}),
            _v("bosque_desenfocado", "Bosque desenfocado", "Forest bokeh",
               "a green forest thrown far out of focus",
               local={"scene": "forest_bokeh", "blur_background": 0.8}),
            _v("hormigon", "Pared de hormigon", "Concrete wall",
               "a plain polished concrete wall", local={"scene": "concrete_wall"}),
            _v("ladrillo", "Pared de ladrillo", "Brick wall",
               "a weathered red brick wall", local={"scene": "brick_wall"}),
            _v("marmol", "Interior de marmol", "Marble interior",
               "a bright marble interior with soft reflections",
               local={"scene": "marble_interior"}),
            _v("cielo_dorado", "Cielo dorado", "Golden sky",
               "an open golden hour sky", local={"scene": "golden_sky"}),
            _v("fondo_oscuro", "Fondo oscuro", "Dark backdrop",
               "a deep charcoal backdrop, low key",
               local={"scene": "dark_backdrop"}),
            _v("lienzo", "Lienzo texturizado", "Textured canvas",
               "a hand painted textured canvas backdrop",
               local={"scene": "textured_canvas"}),
            _v("luz_ventana", "Luz de ventana", "Window light interior",
               "a quiet interior beside a large window",
               local={"scene": "window_interior"}),
        ],
    },
    {
        "group_key": "lighting", "label_es": "Iluminacion", "label_en": "Lighting",
        "multi": True, "sort_order": 70,
        "values": [
            _v("ventana_izq", "Luz de ventana (izquierda)", "Window left",
               "soft directional window light from the left",
               local={"lighting": "window_left"}),
            _v("ventana_der", "Luz de ventana (derecha)", "Window right",
               "soft directional window light from the right",
               local={"lighting": "window_right"}),
            _v("hora_dorada", "Hora dorada", "Golden hour",
               "warm low golden hour sunlight", local={"lighting": "golden_hour"}),
            _v("softbox", "Softbox frontal", "Front softbox",
               "even frontal softbox lighting, gentle shadows",
               local={"lighting": "softbox_front"}),
            _v("contraluz", "Contraluz", "Backlight",
               "backlit with a bright rim around the subject",
               local={"lighting": "rim_backlight"}),
            _v("nublado", "Dia nublado", "Overcast",
               "flat soft overcast daylight", local={"lighting": "overcast"}),
            _v("luz_dura", "Luz dura de estudio", "Hard studio light",
               "a single hard studio light, defined shadows",
               local={"lighting": "hard_key"}),
            _v("difusa", "Luz suave difusa", "Soft diffused",
               "very soft diffused light, almost shadowless",
               local={"lighting": "soft_diffuse"}),
        ],
    },
    {
        "group_key": "framing", "label_es": "Encuadre", "label_en": "Framing",
        "multi": False, "sort_order": 80,
        "values": [
            _v("cuerpo_entero", "Cuerpo entero", "Full body",
               "full body framing, head to feet in frame", shots="full",
               local={"framing": "portrait_full"}),
            _v("medio_cuerpo", "Medio cuerpo", "Half body",
               "waist up framing", local={"framing": "portrait_half"}),
            _v("primer_plano", "Primer plano", "Close up",
               "a close up portrait, head and chest",
               local={"framing": "portrait_closeup"}),
            _v("retrato_cabeza", "Retrato de cabeza", "Headshot",
               "a tight head and shoulders portrait, collarbone and above only",
               local={"framing": "portrait_headshot"}),
            _v("cuadrado", "Cuadrado", "Square",
               "square crop composition", local={"framing": "square"}),
            _v("vertical_9_16", "Vertical 9:16", "Vertical 9:16",
               "tall vertical composition", local={"framing": "story_9x16"}),
        ],
    },
    {
        "group_key": "grade", "label_es": "Acabado de color", "label_en": "Grade",
        "multi": True, "sort_order": 90,
        "values": [
            _v("estudio_neutro", "Estudio neutro", "Neutral studio",
               "neutral true to life colour", local={"grade": "neutral_studio"}),
            _v("film_calido", "Film calido", "Warm film",
               "warm analogue film colour, gentle grain",
               local={"grade": "warm_film"}),
            _v("editorial_frio", "Editorial frio", "Cool editorial",
               "cool desaturated editorial colour",
               local={"grade": "cool_editorial"}),
            _v("blanco_negro", "Blanco y negro", "Black and white",
               "black and white, rich tonal range",
               local={"grade": "editorial_bw"}),
            _v("cine", "Cinematografico", "Cinematic",
               "cinematic teal and orange grade",
               local={"grade": "cinematic_teal_orange"}),
            _v("pastel", "Pastel suave", "Soft pastel",
               "soft pastel palette, lifted blacks",
               local={"grade": "soft_pastel"}),
        ],
    },
    {
        # Deliberately narrow.  Fabric opacity is a legitimate wardrobe choice
        # and also the obvious way to walk a tool towards imagery it must not
        # make, so every value carries a modesty negative and the extremes are
        # simply absent.
        "group_key": "transparency", "label_es": "Tejido",
        "label_en": "Fabric", "multi": False, "sort_order": 100,
        "shot_types": "half,full",
        "values": [
            _v("opaco", "Opaco", "Opaque", "in fully opaque fabric",
               negative="sheer, see through"),
            _v("ligero", "Ligeramente translucido", "Slightly translucent",
               "in a lightly translucent overlay worn over an opaque lining",
               negative="see through, exposed skin, revealing"),
            _v("tejido_fino", "Tejido fino", "Fine fabric",
               "in a fine lightweight fabric, fully lined",
               negative="see through, exposed skin, revealing"),
            _v("gasa", "Gasa", "Chiffon",
               "in layered chiffon over an opaque base",
               negative="see through, exposed skin, revealing"),
        ],
    },
    {
        # A silhouette is a legitimate editorial style, and it is also the only
        # full length frame this engine can build from a reference set that has
        # no clothed full body photograph in it: every pixel of the figure is
        # painted, so nothing of the body shows.  It is a stylisation and the
        # label says so - it is not a picture of the person wearing clothes.
        "group_key": "treatment", "label_es": "Tratamiento",
        "label_en": "Treatment", "multi": False, "sort_order": 95,
        # Only where there is a body in frame; on a headshot it has nothing to do.
        "shot_types": "half,full",
        "values": [
            _v("fotografico", "Fotografico", "Photographic",
               "a natural photograph", local={}),
            _v("silueta", "Silueta a contraluz", "Backlit silhouette",
               "a full length backlit silhouette, the figure rendered as a solid "
               "shape against a bright background, no visible detail on the body",
               local={"treatment": "silhouette"}, shots="half,full"),
        ],
    },
    {
        "group_key": "resolution", "label_es": "Calidad", "label_en": "Quality",
        "multi": False, "sort_order": 110,
        "values": [
            _v("borrador", "Borrador", "Draft", "", params={"quality": "draft"}),
            _v("estandar", "Estandar", "Standard", "",
               params={"quality": "standard"}),
            _v("alta", "Alta", "High", "", params={"quality": "high"}),
            _v("maxima", "Maxima", "Maximum", "", params={"quality": "max"}),
        ],
    },
]

GROUPS_BY_KEY = {g["group_key"]: g for g in BUILTIN_OPTIONS}

# Which groups matter most for each kind of photograph.  This is what makes the
# app "look at the photo and propose what fits" rather than showing one fixed
# menu forever.
_SHOT_PRIORITY = {
    "closeup": ("clothing", "expression", "hair", "lighting", "grade",
                "scene", "clothing_color", "framing"),
    "half": ("clothing", "clothing_color", "expression", "hair", "scene",
             "lighting", "grade", "pose", "framing"),
    "full": ("clothing", "pose", "scene", "clothing_color", "lighting",
             "framing", "hair", "grade", "expression"),
    "unknown": ("clothing", "scene", "lighting", "expression", "hair", "grade"),
}


def groups_for_shot(shot_type: str) -> list[dict]:
    shot = shot_type if shot_type in _SHOT_PRIORITY else "unknown"
    order = _SHOT_PRIORITY[shot]
    out: list[dict] = []
    for group in BUILTIN_OPTIONS:
        allowed = group.get("shot_types", "closeup,half,full").split(",")
        if shot != "unknown" and shot not in allowed:
            continue
        values = [v for v in group["values"]
                  if shot == "unknown" or shot in v["shot_types"].split(",")]
        if not values:
            continue
        copy = dict(group)
        copy["values"] = values
        copy["priority"] = (order.index(group["group_key"])
                            if group["group_key"] in order else 99)
        out.append(copy)
    out.sort(key=lambda g: (g["priority"], g["sort_order"]))
    return out


def suggest_for_analysis(analysis: dict) -> dict:
    """Propose options that make sense for THIS photograph, and say why."""
    analysis = analysis or {}
    shot = str(analysis.get("shot_type") or "unknown")
    quality = analysis.get("quality") or {}
    groups = groups_for_shot(shot)

    suggested: dict[str, list[str]] = {}
    reason: dict[str, str] = {}

    if shot == "closeup":
        suggested["clothing"] = ["blazer_oversize", "camisa_blanca", "jersey_cachemira"]
        reason["clothing"] = ("Aunque sea un primer plano se ve el cuello y los "
                              "hombros, asi que la ropa cambia bastante la foto.")
        suggested["expression"] = ["sonrisa_suave", "mirada_seria"]
        reason["expression"] = ("Es un primer plano: la expresion es lo que mas "
                                "cambia el resultado.")
        suggested["lighting"] = ["ventana_izq", "softbox"]
        reason["lighting"] = "En primeros planos la luz define el rostro."
    elif shot == "full":
        suggested["clothing"] = ["vestido_noche", "traje_sastre", "gabardina"]
        reason["clothing"] = ("Sales de cuerpo entero, asi que la ropa se ve "
                              "completa.")
        suggested["pose"] = ["de_pie_frontal", "tres_cuartos"]
        reason["pose"] = "Con el cuerpo entero la postura cambia mucho la foto."
    else:
        suggested["clothing"] = ["camisa_blanca", "blazer_oversize", "top_punto"]
        reason["clothing"] = "Es un plano medio: se ve la ropa de cintura arriba."
        suggested["expression"] = ["sonrisa_suave"]
        reason["expression"] = "La cara sigue siendo protagonista."

    exposure = float(quality.get("exposure") or 0.5)
    if exposure < 0.35:
        suggested["lighting"] = ["softbox", "difusa"]
        reason["lighting"] = ("La foto original tiene poca luz, conviene una "
                              "iluminacion mas clara.")
    elif exposure > 0.8:
        suggested["lighting"] = ["nublado", "difusa"]
        reason["lighting"] = "La foto original esta muy clara, suavizamos la luz."

    suggested.setdefault("scene", ["estudio_gris", "bokeh_calido"])
    reason.setdefault("scene", "Fondos que funcionan bien con esta foto.")
    suggested.setdefault("grade", ["estudio_neutro"])
    reason.setdefault("grade", "Acabado neutro para mantener tu tono de piel real.")

    return {"groups": groups, "suggested": suggested, "reason": reason,
            "shot_type": shot}


def resolve_choices(choices: dict, shot_type: str) -> dict:
    """Drop anything that is not in the catalogue for this kind of photograph."""
    valid = {g["group_key"]: {v["value_key"] for v in g["values"]}
             for g in groups_for_shot(shot_type)}
    out: dict[str, list[str]] = {}
    for group, values in (choices or {}).items():
        allowed = valid.get(group)
        if not allowed:
            continue
        if not isinstance(values, (list, tuple, set)):
            values = [values]
        kept = [str(v) for v in values if str(v) in allowed]
        if kept:
            out[group] = kept
    return out


def value_of(group_key: str, value_key: str) -> dict[str, Any] | None:
    group = GROUPS_BY_KEY.get(group_key)
    if not group:
        return None
    for value in group["values"]:
        if value["value_key"] == value_key:
            return value
    return None


# ------------------------------------------------------------ garment lookup

# The classification above, keyed by value.  It is read from the catalogue row
# first (``params.garment``, written by seed.py) and only falls back to this
# table when a caller hands over a bare value key or an options table that has
# not been reseeded yet.
def _garment_index() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for group in BUILTIN_OPTIONS:
        for value in group["values"]:
            info = (value.get("params") or {}).get("garment")
            if info:
                out[value["value_key"]] = dict(info)
    return out


GARMENT_BY_VALUE: dict[str, dict] = _garment_index()

# A user can add her own garment to the wardrobe - options rows carry a
# user_id and the admin screen writes them - and those rows have no
# classification at all.  Reading the words of the fragment is a guess, and it
# is reported as one, but a guess that answers "trousers" to a fragment that
# says trousers beats the alternative that was in place until today, which was
# to say nothing and let the engine keep whatever the photograph had on.
_LOWER_WORDS = ("trouser", "pant", "jean", "skirt", "legging", "short",
                "pantalon", "vaquero", "falda", "chino", "culotte", "bermuda")
_DRESS_WORDS = ("dress", "gown", "jumpsuit", "vestido", "mono", "romper",
                "playsuit", "overall", "kaftan")
_UPPER_WORDS = ("shirt", "blouse", "top", "sweater", "jumper", "knit",
                "blazer", "jacket", "coat", "cardigan", "camisa", "blusa",
                "jersey", "chaqueta", "abrigo", "gabardina", "camiseta",
                "sudadera", "corset", "bodysuit", "kimono", "poncho")

# Used only when a garment is known to be a top and declares no bottom of its
# own.  Full length on purpose: "trousers" alone still lets the engine stop the
# garment at the hip, which is one word away from the failure being fixed.
DEFAULT_BOTTOM = "dark tailored trousers, full length to the ankle"

# The mirror of DEFAULT_BOTTOM, and it exists because the same measurement that
# justified the trousers was run in the other direction.  Sweeping every one and
# two garment selection the wardrobe allows (600 prompts, 24 options, framings
# half and full) the lower half is now named in 600 of 600 - but the UPPER half
# is named in only 558: the 21 selections made only of the six bottoms leave the
# chest unnamed while the same request still says "dress her in this outfit and
# in nothing else" and drops the clause that preserved the source clothing.  On
# a photograph of a person in lingerie that is an instruction to undress her,
# the exact mirror of the delivered image that started all this.  Neutral and
# fully covering on purpose: it has to be wearable under any of the six bottoms
# and must not become a second styling decision.
DEFAULT_TOP = ("a plain long sleeve top in a neutral tone, fully covering the "
               "chest and shoulders, tucked in at the waistband")


def _word(text: str, roots) -> bool:
    """Match at the start of a word, not anywhere inside one.

    A plain substring test reads "kimono" as "mono" and files a silk kimono as
    a jumpsuit, which is how a top ends up classified as a complete outfit and
    loses its trousers.  Matching the start of a word keeps the plural ("root"
    finds "trousers") without inventing the stem in the middle of another word.
    """
    return any(re.search(r"\b" + re.escape(root), text) for root in roots)


def _garment_guess(text: str) -> dict:
    words = text.lower()
    has_lower = _word(words, _LOWER_WORDS)
    has_upper = _word(words, _UPPER_WORDS)
    if _word(words, _DRESS_WORDS):
        return {"kind": "complete", "lower": "dress", "bottom": "",
                "source": "deducido"}
    if has_lower and has_upper:
        return {"kind": "complete", "lower": "trousers", "bottom": "",
                "source": "deducido"}
    if has_lower:
        lower = "skirt" if ("skirt" in words or "falda" in words) else "trousers"
        return {"kind": "bottom", "lower": lower, "bottom": "",
                "source": "deducido"}
    if has_upper:
        return {"kind": "top", "lower": "trousers", "bottom": DEFAULT_BOTTOM,
                "source": "deducido"}
    # Nothing recognisable.  Saying "top" here would put trousers on a garment
    # nobody described, so this abstains: the caller still forbids visible
    # underwear and still says replace rather than layer, it just does not
    # invent a lower half.
    return {"kind": "unknown", "lower": "", "bottom": "", "source": "sin clasificar"}


def garment_info(option: Any) -> dict:
    """What this wardrobe choice covers: ``kind``, ``lower``, ``bottom``.

    Accepts a bare value key, a catalogue row, or the resolved option shape
    that ``generation.prompt`` passes around.  Always returns a dict, so the
    caller never has to branch on missing data.
    """
    key = ""
    params: dict = {}
    text_bits: list[str] = []
    if isinstance(option, str):
        key = option.strip()
    elif isinstance(option, dict):
        key = str(option.get("value") or option.get("value_key")
                  or option.get("key") or "").strip()
        raw = option.get("params")
        if isinstance(raw, dict):
            params = raw
        for field in ("prompt", "prompt_fragment", "label_en", "label",
                      "label_es"):
            value = option.get(field)
            if value:
                text_bits.append(str(value))
    declared = params.get("garment")
    if isinstance(declared, dict) and declared.get("kind"):
        info = {"kind": str(declared.get("kind") or "top"),
                "lower": str(declared.get("lower") or ""),
                "bottom": str(declared.get("bottom") or ""),
                "source": "catalogo"}
        return info
    known = GARMENT_BY_VALUE.get(key)
    if known:
        info = dict(known)
        info.setdefault("lower", "")
        info.setdefault("bottom", "")
        info["source"] = "catalogo"
        return info
    return _garment_guess(" ".join(text_bits) or key.replace("_", " "))

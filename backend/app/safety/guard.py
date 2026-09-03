"""What this system will not generate, and why the rule lives in code.

The developer already told the client, in writing, where the line is: this tool
does fashion, editorial, portrait, product and scenery from her real
photographs, and it does not produce intimate or sexualised imagery of a real,
identifiable person.  The reasoning he gave is the reasoning encoded here - the
same machinery that would do it for someone who consented does it just as
happily for someone who did not, and the profile system is explicitly built to
work with many different people.

So the boundary is enforced by the program rather than remembered by a person.
A future maintainer reading this: removing this check does not add a feature, it
removes the one thing that keeps a general identity-preserving image tool from
being an efficient way to do that to strangers.

What this is NOT: a filter on the user's own reference photographs.  What
someone keeps in their own private album is their business, and check_upload
flags rather than refuses, because the profile needs those photographs to
measure a body correctly.  The line is on the OUTPUT.
"""
from __future__ import annotations

import re
import unicodedata

# Spanish, Portuguese and English: the client writes in the first two.
_BLOCK_TERMS = (
    # intimate clothing / undress
    "lenceria", "lingerie", "ropa interior", "roupa interior", "underwear",
    "bra", "sujetador", "sutia", "brasier", "panties", "bragas", "calcinha",
    "tanga", "thong", "corset", "corse", "espartilho", "body de encaje",
    "babydoll", "negligee", "camison transparente",
    "desnuda", "desnudo", "desnudez", "nua", "nu", "nude", "naked", "topless",
    "sin ropa", "sem roupa", "sin sujetador", "braless",
    "pechos al aire", "senos al aire", "seios de fora", "bare breasts",
    "nipples", "pezones", "mamilos", "areola",
    "entrepierna", "genitales", "genitals", "vagina", "vulva", "pubis",
    # bath / shower framings the client asked about explicitly
    "banera", "bañera", "banheira", "bathtub", "en la bañera", "na banheira",
    "ducha desnuda", "shower nude", "jacuzzi desnuda",
    # sexualisation
    "erotico", "erotica", "erotico", "erotic", "sensual explicita",
    "sexy explicito", "sexual", "porn", "porno", "pornografico", "nsfw",
    "onlyfans", "fetiche", "fetish", "bdsm", "seductora desnuda",
    "provocativa desnuda",
)

# Terms that are fine on their own but not when applied to a person's body.
_CONTEXT_TERMS = (("transparente", ("ropa", "vestido", "blusa", "tejido",
                                    "camisa", "top", "falda")),)

_MINOR_TERMS = (
    "nina", "nino", "niña", "niño", "menor", "menores", "adolescente",
    "child", "kid", "teen", "teenager", "minor", "underage", "colegiala",
    "schoolgirl", "criança", "crianca", "bebe", "baby", "infantil",
    "loli", "preadolescente",
)

# Naming a third party rather than the profile owner.
_THIRD_PARTY = (
    "famosa", "famoso", "celebrity", "celebridad", "actriz", "actor",
    "cantante", "singer", "modelo famosa", "influencer", "presidenta",
    "presidente", "mi amiga", "mi vecina", "my friend", "my neighbour",
    "minha amiga", "sin su permiso", "sem permissao", "without her consent",
    "without permission", "no lo sabe", "nao sabe",
)

_INTIMATE_OPTION_KEYS = {
    "lenceria", "lingerie", "ropa_interior", "bikini_intimo", "banera",
    "bathtub", "desnudo", "nude", "topless", "transparente_total",
}


def _normalise(text: str) -> str:
    """Lowercase, strip accents, collapse spacing: 'Lencería' == 'lenceria'."""
    raw = unicodedata.normalize("NFKD", str(text or "").lower())
    raw = "".join(c for c in raw if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", raw).strip()


def _hits(text: str, terms) -> list[str]:
    found = []
    for term in terms:
        pattern = r"(?<!\w)" + re.escape(_normalise(term)) + r"(?!\w)"
        if re.search(pattern, text):
            found.append(term)
    return found


def is_intimate_request(text: str) -> bool:
    return bool(_hits(_normalise(text), _BLOCK_TERMS))


def _collect_text(brief: dict, options: dict) -> str:
    parts: list[str] = []
    for key in ("prompt", "instructions", "notes", "description", "extra",
                "clothing", "setting", "pose", "style", "free_text"):
        value = (brief or {}).get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (list, tuple)):
            parts.extend(str(v) for v in value)
    for group, values in (options or {}).items():
        parts.append(str(group))
        if isinstance(values, (list, tuple, set)):
            parts.extend(str(v) for v in values)
        else:
            parts.append(str(values))
    return _normalise(" ".join(parts))


def check_request(brief: dict, options: dict, profile: dict, user: dict) -> dict:
    """Decide whether this generation may run.  Called before any money moves."""
    text = _collect_text(brief, options)

    minors = _hits(text, _MINOR_TERMS)
    if minors:
        return {
            "allowed": False, "code": "minor",
            "reason": ("Esta peticion menciona a una persona menor de edad. El "
                       "sistema solo trabaja con fotografias de personas adultas "
                       "que han dado su consentimiento."),
            "blocked_terms": minors,
        }

    intimate = _hits(text, _BLOCK_TERMS)
    for term, needed in _CONTEXT_TERMS:
        if term in text and not any(word in text for word in needed):
            intimate.append(term)
    if intimate:
        return {
            "allowed": False, "code": "intimate",
            "reason": ("Este sistema no genera imagenes intimas ni de contenido "
                       "sexual de personas reales. Si puede hacer moda, "
                       "editorial, retrato, producto y escenarios, manteniendo "
                       "tu cuerpo, tu piel y tus tatuajes tal y como son."),
            "blocked_terms": sorted(set(intimate)),
        }

    chosen_keys = set()
    for values in (options or {}).values():
        if isinstance(values, (list, tuple, set)):
            chosen_keys.update(_normalise(v) for v in values)
        else:
            chosen_keys.add(_normalise(values))
    bad_keys = sorted(chosen_keys & _INTIMATE_OPTION_KEYS)
    if bad_keys:
        return {
            "allowed": False, "code": "intimate_option",
            "reason": ("Alguna de las opciones elegidas corresponde a imagen "
                       "intima, que este sistema no genera."),
            "blocked_terms": bad_keys,
        }

    third = _hits(text, _THIRD_PARTY)
    if third:
        return {
            "allowed": False, "code": "third_party",
            "reason": ("La peticion parece referirse a otra persona. Cada perfil "
                       "solo puede generar imagenes de la persona que dio su "
                       "consentimiento para ese perfil."),
            "blocked_terms": third,
        }

    return {"allowed": True, "reason": "", "code": "", "blocked_terms": []}


def check_upload(image_path: str) -> dict:
    """Look at a reference photograph.  Flags, never refuses.

    A woman's own private photographs are hers.  The profile builder needs them
    to measure a real body rather than guess one, so an upload is not rejected
    for being intimate - it is marked, so the interface can remind her that this
    class of image will not be generated, and so a reviewer can see the flag.
    """
    flags: list[str] = []
    try:
        from ..analysis import loader, pose as pose_mod
        from ..analysis import skin as skin_mod
        from ..analysis import face as face_mod

        img = loader.load_image(image_path, max_side=1024)
        pose_d = pose_mod.detect_pose(img)
        face_d = face_mod.detect_face(img)
        skin_d = skin_mod.skin_stats(img, pose_d, face_d)
        # A very high proportion of exposed skin on the torso is the only
        # signal used, and it only ever sets a flag.
        exposure = float(skin_d.get("torso_skin_fraction") or 0.0)
        if exposure > 0.65:
            flags.append("mucha_piel_visible")
    except Exception:                                    # noqa: BLE001
        return {"allowed": True, "reason": "", "flags": []}

    reason = ""
    if flags:
        reason = ("Esta foto se guardara como referencia para medir tus "
                  "proporciones. El sistema no genera imagenes intimas.")
    return {"allowed": True, "reason": reason, "flags": flags}

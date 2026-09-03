"""Consent records, one per person profile.

The client's stated plan is to use the system on other people later, which is
exactly when "she said it was fine" stops being good enough.  A profile whose
subject is someone other than the account holder cannot generate anything until
a consent record exists saying who granted it, when, and for what.

Revocation is soft on purpose: the record stays as evidence that permission once
existed and was withdrawn, which is more useful than a deleted row.
"""
from __future__ import annotations

from .. import db

VALID_RELATIONSHIPS = ("self", "client")
DEFAULT_SCOPE = ("moda", "editorial", "retrato", "escenarios", "producto")


def record_consent(user_id: str, profile_id: str, payload: dict) -> dict:
    profile = db.row_to_dict(
        db.q1("SELECT * FROM profiles WHERE id=? AND user_id=?",
              (profile_id, user_id)))
    if not profile:
        raise ValueError("Ese perfil no existe.")

    relationship = str(payload.get("relationship") or "self").lower()
    if relationship not in VALID_RELATIONSHIPS:
        raise ValueError("La relacion debe ser 'self' o 'client'.")
    granted_by = str(payload.get("granted_by") or "").strip()
    if relationship == "client" and not granted_by:
        raise ValueError("Indica el nombre de la persona que da su permiso.")

    record = {
        "granted_by": granted_by or "titular de la cuenta",
        "relationship": relationship,
        "statement": str(payload.get("statement") or "").strip()
        or ("Autorizo el uso de mis fotografias para generar imagenes de moda, "
            "editorial, retrato y escenarios con este sistema."),
        "signed_at": db.now(),
        "ip": str(payload.get("ip") or ""),
        "evidence_note": str(payload.get("evidence_note") or ""),
        "scope": list(payload.get("scope") or DEFAULT_SCOPE),
        "revoked_at": None,
        "revoked_reason": "",
    }
    db.execute("UPDATE profiles SET consent_json=?, updated_at=? WHERE id=?",
               (db.dumps(record), db.now(), profile_id))
    db.audit("consent.record", user_id, profile_id=profile_id,
             relationship=relationship)
    return record


def get_consent(profile_id: str) -> dict:
    row = db.q1("SELECT consent_json FROM profiles WHERE id=?", (profile_id,))
    return db.loads(row["consent_json"], {}) if row else {}


def has_valid_consent(profile_id: str) -> bool:
    """Self-portraits are covered by the account itself; anyone else is not."""
    consent = get_consent(profile_id)
    if consent.get("revoked_at"):
        return False
    relationship = str(consent.get("relationship") or "")
    if relationship == "self":
        return True
    if relationship == "client":
        return bool(consent.get("granted_by") and consent.get("signed_at"))
    # No record at all: allowed only when the profile is the account owner's own.
    row = db.q1("SELECT is_default FROM profiles WHERE id=?", (profile_id,))
    return bool(row and int(row["is_default"] or 0) == 1)


def consent_problem(profile_id: str) -> str:
    """Empty string when fine, otherwise the Spanish reason to show the user."""
    if has_valid_consent(profile_id):
        return ""
    consent = get_consent(profile_id)
    if consent.get("revoked_at"):
        return ("El permiso de esta persona fue retirado, no se pueden generar "
                "mas imagenes de este perfil.")
    return ("Este perfil todavia no tiene registrado el consentimiento de la "
            "persona. Registralo antes de generar imagenes.")


def revoke(profile_id: str, reason: str) -> None:
    consent = get_consent(profile_id)
    consent["revoked_at"] = db.now()
    consent["revoked_reason"] = str(reason or "")
    db.execute("UPDATE profiles SET consent_json=?, updated_at=? WHERE id=?",
               (db.dumps(consent), db.now(), profile_id))
    db.audit("consent.revoke", None, profile_id=profile_id, reason=reason)

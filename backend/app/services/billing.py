"""Money: balances, spending limits and the warnings that come before zero.

The client was explicit about three things, and this module exists to honour
them literally rather than approximately:

  1. warn her *while there is still balance*, saying how much is left and
     roughly how much to top up;
  2. when the balance cannot cover the next image, **stop** and tell her at
     once, instead of carrying on and failing image after image;
  3. never, under any circumstance, charge her card automatically.

Point three is why there is no payment code anywhere in this file.  ``recharge``
records money the user has already added on the provider's own website; it moves
nothing.  The ledger here mirrors the provider's balance so the app can reason
about it offline - the authoritative number always lives on the provider's own
dashboard, and the UI says so.
"""
from __future__ import annotations

import threading
import time

from .. import db
from ..config import SETTINGS

# Rough price of one finished image per quality tier, used for estimates and
# for translating a balance into "about N more photos".
TIER_PRICE_USD = {
    "draft": 0.0,
    "preview": 0.012,
    "standard": 0.035,
    "high": 0.055,
    "max": 0.09,
}

# Providers that cost nothing never gate anything.
FREE_PROVIDERS = {"local", "heuristic", ""}

ALERT_COOLDOWN_S = 12 * 3600


# ------------------------------------------------------------------ balances

def balance(user_id: str, provider: str) -> float:
    if provider in FREE_PROVIDERS:
        return float("inf")
    row = db.q1(
        "SELECT COALESCE(SUM(amount_usd), 0) AS total FROM ledger "
        "WHERE user_id=? AND provider=?",
        (user_id, provider),
    )
    return round(float(row["total"]) if row else 0.0, 6)


def spent_since(user_id: str, provider: str | None, since: float) -> float:
    sql = ("SELECT COALESCE(SUM(-amount_usd), 0) AS total FROM ledger "
           "WHERE user_id=? AND kind='spend' AND created_at >= ?")
    params: list = [user_id, since]
    if provider:
        sql += " AND provider=?"
        params.append(provider)
    row = db.q1(sql, params)
    return round(float(row["total"]) if row else 0.0, 6)


def _day_start() -> float:
    now = time.localtime()
    return time.mktime((now.tm_year, now.tm_mon, now.tm_mday, 0, 0, 0, 0, 0, -1))


def _month_start() -> float:
    now = time.localtime()
    return time.mktime((now.tm_year, now.tm_mon, 1, 0, 0, 0, 0, 0, -1))


def all_balances(user_id: str) -> dict:
    out: dict[str, dict] = {}
    for provider in ("anthropic", "fal"):
        bal = balance(user_id, provider)
        out[provider] = {
            "balance": bal,
            "spent_today": spent_since(user_id, provider, _day_start()),
            "spent_30d": spent_since(user_id, provider, time.time() - 30 * 86400),
            "status": _status(bal),
            "photos_left": photos_left(bal, "standard"),
        }
    out["local"] = {"balance": None, "spent_today": 0.0, "spent_30d": 0.0,
                    "status": "free", "photos_left": None}
    return out


def _status(bal: float) -> str:
    limits = SETTINGS.limits
    if bal <= 0:
        return "zero"
    if bal < limits.critical_balance_usd:
        return "critical"
    if bal < limits.low_balance_usd:
        return "low"
    return "ok"


def photos_left(bal: float, quality: str = "standard") -> int | None:
    price = TIER_PRICE_USD.get(quality, TIER_PRICE_USD["standard"])
    if price <= 0:
        return None
    return max(0, int(bal / price))


def price_of_next_image(user_id: str, quality: str) -> float:
    return float(TIER_PRICE_USD.get(quality, TIER_PRICE_USD["standard"]))


def recommended_topup(user_id: str, provider: str) -> float:
    """A number the user can act on, based on what she actually spends.

    Thirty days of history, projected forward a month, rounded up to something
    a human would type into a top-up box.
    """
    spent = spent_since(user_id, provider, time.time() - 30 * 86400)
    target = max(5.0, spent * 1.2)
    for step in (5.0, 10.0, 20.0, 50.0, 100.0):
        if target <= step:
            return step
    return round(target / 10.0) * 10.0


# ------------------------------------------------------------------- spending

def can_spend(user_id: str, provider: str, amount_usd: float) -> dict:
    """The gate the orchestrator calls before every single provider call."""
    return _gate(user_id, provider, amount_usd, 0.0, 0.0)


def _gate(user_id: str, provider: str, amount_usd: float,
          held_provider: float, held_user: float) -> dict:
    """``can_spend`` with the money already promised to calls still in flight.

    ``held_provider`` is what this provider's open reservations will cost, and
    it comes off the balance; ``held_user`` is the same figure across every
    provider and it comes off the daily and monthly caps, because those count
    the whole account.  Both are zero for the plain sequential gate, so
    ``can_spend`` answers exactly as it always did, down to the wording.
    """
    amount = max(0.0, float(amount_usd or 0.0))
    if provider in FREE_PROVIDERS or amount <= 0:
        return {"ok": True, "reason": "", "balance": None,
                "remaining_daily": None, "remaining_monthly": None, "alert": None}

    user = db.row_to_dict(db.q1("SELECT * FROM users WHERE id=?", (user_id,))) or {}
    # Held money is spoken for even though the ledger has not seen it yet, so
    # every number below is what is really left to promise to one more call.
    bal = round(balance(user_id, provider) - max(0.0, float(held_provider)), 6)
    daily_cap = float(user.get("daily_limit_usd") or SETTINGS.limits.default_daily_usd)
    monthly_cap = float(user.get("monthly_limit_usd")
                        or SETTINGS.limits.default_monthly_usd)
    spent_day = spent_since(user_id, None, _day_start()) + max(0.0, float(held_user))
    spent_month = spent_since(user_id, None, _month_start()) + max(0.0, float(held_user))
    remaining_daily = round(daily_cap - spent_day, 6)
    remaining_monthly = round(monthly_cap - spent_month, 6)

    base = {"balance": bal, "remaining_daily": remaining_daily,
            "remaining_monthly": remaining_monthly}

    if bal - amount < 0:
        topup = recommended_topup(user_id, provider)
        return {**base, "ok": False,
                "reason": ("Te has quedado sin saldo en %s. Quedan %.2f USD y esta "
                           "imagen cuesta %.2f USD. Recarga unos %.0f USD para "
                           "seguir." % (provider, bal, amount, topup)),
                "alert": {"kind": "zero_balance", "level": "critical",
                          "provider": provider, "topup": topup, "balance": bal}}
    if remaining_daily - amount < 0:
        return {**base, "ok": False,
                "reason": ("Has alcanzado tu limite de gasto de hoy (%.2f USD). "
                           "Puedes subirlo en Ajustes." % daily_cap),
                "alert": {"kind": "limit_reached", "level": "warning",
                          "provider": provider, "scope": "diario", "cap": daily_cap}}
    if remaining_monthly - amount < 0:
        return {**base, "ok": False,
                "reason": ("Has alcanzado tu limite de gasto del mes (%.2f USD). "
                           "Puedes subirlo en Ajustes." % monthly_cap),
                "alert": {"kind": "limit_reached", "level": "warning",
                          "provider": provider, "scope": "mensual",
                          "cap": monthly_cap}}
    return {**base, "ok": True, "reason": "", "alert": None}


def charge(user_id: str, provider: str, amount_usd: float, ref: str,
           note: str = "") -> float:
    """Record real money spent.  Returns the balance afterwards."""
    amount = float(amount_usd or 0.0)
    if provider in FREE_PROVIDERS or amount <= 0:
        return balance(user_id, provider)
    after = round(balance(user_id, provider) - amount, 6)
    db.execute(
        "INSERT INTO ledger(id,user_id,provider,kind,amount_usd,balance_after,"
        "ref,note,created_at) VALUES(?,?,?,'spend',?,?,?,?,?)",
        (db.new_id("led"), user_id, provider, -amount, after, ref, note, db.now()),
    )
    check_and_raise_alerts(user_id)
    return after


# --------------------------------------------------------------- reservations
#
# Sequential code could ask "can I afford this?" and then spend, because
# nothing else moved in between.  The robot now sends several images of the
# same run to the provider at once, and three workers asking that question at
# the same instant would all be told "yes" about the same last dollar - and all
# three would spend it.  A reservation closes that window: the estimated price
# is held under a lock before the call leaves, every later gate sees the held
# money as already gone, and when the call comes back the hold becomes the real
# charge or is simply dropped.  So the total spent can never exceed what the
# plain sequential gate would have allowed.
#
# Holds live in memory, not in the ledger: the ledger is real money and an
# estimate is not.  One process owns them, which is how the application runs;
# a restart loses nothing, because a lost hold only ever frees money that was
# never spent.

_holds_lock = threading.Lock()
_holds: dict[str, dict] = {}


def _held_locked(user_id: str, provider: str | None) -> float:
    """Money promised to calls in flight.  Caller must hold ``_holds_lock``."""
    total = 0.0
    for hold in _holds.values():
        if hold["user_id"] != user_id:
            continue
        if provider is not None and hold["provider"] != provider:
            continue
        total += hold["amount"]
    return round(total, 6)


def reserve(user_id: str, provider: str, amount_usd: float,
            ref: str = "") -> dict:
    """Gate a call that is about to be sent, and hold its estimated cost.

    The answer is ``can_spend``'s, so a refusal carries the same reason and the
    same alert and must be treated the same way: stop the whole run.  On a yes
    the caller owns ``hold_id`` and MUST end it exactly once, with ``settle``
    when the provider charged, or ``release`` when it did not - a stranded hold
    would keep money invisible until the process restarts.
    """
    amount = max(0.0, float(amount_usd or 0.0))
    with _holds_lock:
        gate = _gate(user_id, provider, amount,
                     _held_locked(user_id, provider),
                     _held_locked(user_id, None))
        if not gate["ok"]:
            return {**gate, "hold_id": "", "amount": amount}
        hold_id = db.new_id("hold")
        _holds[hold_id] = {"user_id": user_id, "provider": provider,
                           "amount": amount, "ref": ref, "at": time.time()}
    return {**gate, "hold_id": hold_id, "amount": amount}


def release(hold_id: str) -> float:
    """End a hold without spending.  Returns the amount that was freed."""
    with _holds_lock:
        hold = _holds.pop(hold_id, None)
    return float(hold["amount"]) if hold else 0.0


def settle(user_id: str, provider: str, hold_id: str, amount_usd: float,
           ref: str, note: str = "") -> float:
    """Turn a hold into the charge the provider actually made.

    Dropping the hold and writing the ledger row happen under the same lock,
    and that is the whole point: in between, the money is promised by nothing
    and spent by nothing, so a gate running just then would see it as free and
    hand the same dollar to a second call.  Held under the lock, no gate can
    observe that gap - the money goes straight from promised to spent.  The
    real cost is what reaches the ledger, exactly as an ungated ``charge``
    would have written it.
    """
    with _holds_lock:
        _holds.pop(hold_id, None)
        return charge(user_id, provider, amount_usd, ref, note)


def held(user_id: str, provider: str | None = None) -> float:
    """What this user has promised to calls that have not answered yet."""
    with _holds_lock:
        return _held_locked(user_id, provider)


def recharge(user_id: str, provider: str, amount_usd: float, note: str = "") -> dict:
    """Write down money the user already added at the provider's own website.

    This function deliberately talks to no payment system.  Nothing here can
    charge a card, and the response says so, because the client asked to be the
    only one who decides when and how much to top up.
    """
    amount = float(amount_usd or 0.0)
    if amount <= 0:
        raise ValueError("El importe debe ser mayor que cero.")
    if amount > 1000:
        raise ValueError("Importe demasiado alto. Registra recargas reales.")
    after = round(balance(user_id, provider) + amount, 6)
    db.execute(
        "INSERT INTO ledger(id,user_id,provider,kind,amount_usd,balance_after,"
        "ref,note,created_at) VALUES(?,?,?,'recharge',?,?,'manual',?,?)",
        (db.new_id("led"), user_id, provider, amount, after, note, db.now()),
    )
    db.audit("billing.recharge", user_id, provider=provider, amount=amount)
    # Clear the standing "no balance" alerts: the situation has changed.
    db.execute(
        "UPDATE alerts SET read_at=? WHERE user_id=? AND read_at IS NULL "
        "AND kind IN ('zero_balance','low_balance')",
        (db.now(), user_id),
    )
    return {
        "provider": provider, "amount_usd": amount, "balance": after,
        "photos_left": photos_left(after, "standard"),
        "message": ("Anotado. Este registro solo refleja el saldo que ya has "
                    "anadido en la web de %s: la aplicacion nunca cobra nada por "
                    "su cuenta." % provider),
    }


# --------------------------------------------------------------------- alerts

def raise_alert(user_id: str, kind: str, level: str, title: str, message: str,
                **payload) -> dict | None:
    """One alert of a kind per provider per cooldown, so it never becomes noise."""
    provider = str(payload.get("provider") or "")
    recent = db.q1(
        "SELECT id FROM alerts WHERE user_id=? AND kind=? AND created_at > ? "
        "AND payload_json LIKE ?",
        (user_id, kind, db.now() - ALERT_COOLDOWN_S, f'%"{provider}"%'),
    )
    if recent:
        return None
    alert_id = db.new_id("alr")
    db.execute(
        "INSERT INTO alerts(id,user_id,kind,level,title,message,payload_json,"
        "created_at) VALUES(?,?,?,?,?,?,?,?)",
        (alert_id, user_id, kind, level, title, message, db.dumps(payload), db.now()),
    )
    return {"id": alert_id, "kind": kind, "level": level, "title": title,
            "message": message, "payload": payload}


def check_and_raise_alerts(user_id: str) -> list[dict]:
    """Warn before zero, not at zero.  Called after every charge."""
    limits = SETTINGS.limits
    raised: list[dict] = []
    for provider in ("anthropic", "fal"):
        bal = balance(user_id, provider)
        if bal <= 0:
            continue                       # zero is reported by the spend gate
        topup = recommended_topup(user_id, provider)
        left = photos_left(bal, "standard")
        if bal < limits.critical_balance_usd:
            alert = raise_alert(
                user_id, "low_balance", "critical",
                "Saldo casi agotado en %s" % provider,
                "Te quedan %.2f USD en %s, para unas %d fotos mas. Recarga unos "
                "%.0f USD cuando puedas." % (bal, provider, left or 0, topup),
                provider=provider, balance=bal, topup=topup, photos_left=left)
        elif bal < limits.low_balance_usd:
            alert = raise_alert(
                user_id, "low_balance", "warning",
                "Saldo bajo en %s" % provider,
                "Te quedan %.2f USD en %s, para unas %d fotos mas. Cuando bajes de "
                "ahi el robot se detendra y te avisara."
                % (bal, provider, left or 0),
                provider=provider, balance=bal, topup=topup, photos_left=left)
        else:
            alert = None
        if alert:
            raised.append(alert)
    return raised


def unread_alerts(user_id: str, limit: int = 50) -> list[dict]:
    rows = db.q(
        "SELECT * FROM alerts WHERE user_id=? AND read_at IS NULL "
        "ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    )
    return db.rows_to_dicts(rows)


# ---------------------------------------------------------------------- usage

def usage(user_id: str, days: int = 30) -> dict:
    """The numbers the settings page shows, including the headline metric.

    "Intentos por foto conseguida" is the one the client actually cares about:
    it is the manual-versus-robot comparison expressed as a single number.
    """
    since = time.time() - max(1, int(days)) * 86400
    att = db.q1(
        "SELECT COUNT(*) AS n, "
        "SUM(CASE WHEN status='accepted' THEN 1 ELSE 0 END) AS accepted, "
        "SUM(CASE WHEN status='rejected' THEN 1 ELSE 0 END) AS rejected, "
        "SUM(CASE WHEN status='repaired' THEN 1 ELSE 0 END) AS repaired, "
        "COALESCE(SUM(cost_usd),0) AS cost "
        "FROM attempts WHERE user_id=? AND created_at >= ?",
        (user_id, since),
    )
    total = int(att["n"] or 0) if att else 0
    accepted = int(att["accepted"] or 0) if att else 0
    cost = float(att["cost"] or 0.0) if att else 0.0

    per_provider = db.rows_to_dicts(db.q(
        "SELECT provider, COUNT(*) AS n, COALESCE(SUM(cost_usd),0) AS cost "
        "FROM attempts WHERE user_id=? AND created_at >= ? GROUP BY provider",
        (user_id, since),
    ))
    trend = db.rows_to_dicts(db.q(
        "SELECT date(created_at,'unixepoch','localtime') AS day, "
        "COUNT(*) AS count, COALESCE(SUM(cost_usd),0) AS cost "
        "FROM attempts WHERE user_id=? AND created_at >= ? "
        "GROUP BY day ORDER BY day",
        (user_id, since),
    ))
    reasons = db.rows_to_dicts(db.q(
        "SELECT reject_reason AS reason, COUNT(*) AS count FROM attempts "
        "WHERE user_id=? AND created_at >= ? AND status='rejected' "
        "AND reject_reason <> '' GROUP BY reason ORDER BY count DESC LIMIT 10",
        (user_id, since),
    ))
    return {
        "days": int(days),
        "attempts": total,
        "accepted": accepted,
        "rejected": int(att["rejected"] or 0) if att else 0,
        "repaired": int(att["repaired"] or 0) if att else 0,
        "attempts_per_photo": round(total / accepted, 2) if accepted else None,
        "total_usd": round(cost, 4),
        "cost_per_photo_usd": round(cost / accepted, 4) if accepted else None,
        "by_provider": per_provider,
        "trend": trend,
        "reject_reasons": reasons,
        "balances": all_balances(user_id),
    }

"""FastAPI application factory.

Serves both the JSON API under /api and the static mobile web app at /.
Everything is one process on purpose: the client runs this on a single small
machine and must be able to open it from an iPhone without any build step.
"""
from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import db
from .config import FRONTEND_DIR, LOG_DIR, SETTINGS, ensure_dirs

log = logging.getLogger("photorobot")


def _setup_logging() -> None:
    ensure_dirs()
    fmt = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8"))
    except OSError:
        pass
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def create_app() -> FastAPI:
    _setup_logging()
    ensure_dirs()
    db.init_db()

    app = FastAPI(
        title=SETTINGS.app_name,
        version=SETTINGS.version,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------- routers
    from .routers import (admin, album, auth, catalog, favorites, files,
                          generate, originals, profiles, settings)

    for module in (auth, profiles, originals, catalog, generate, album,
                   favorites, settings, admin, files):
        app.include_router(module.router, prefix="/api")

    # ------------------------------------------------------------- startup
    @app.on_event("startup")
    def _startup() -> None:
        from .catalog import seed
        from .services import billing  # noqa: F401  (registers nothing, warms import)
        seed.seed_all()
        from .providers import registry
        avail = registry.availability()
        log.info("Providers: %s", ", ".join(
            f"{k}={'on' if v['available'] else 'off'}" for k, v in avail.items()))
        log.info("%s v%s ready - data at %s", SETTINGS.app_name, SETTINGS.version,
                 db.DB_PATH.parent)

    # --------------------------------------------------------- error shape
    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.error("Unhandled error on %s %s\n%s", request.method, request.url.path,
                  traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": "Error interno del servidor. Revisa los registros."},
        )

    # -------------------------------------------------------------- health
    @app.get("/api/health")
    def health() -> dict:
        from .providers import registry
        return {
            "ok": True,
            "app": SETTINGS.app_name,
            "version": SETTINGS.version,
            "python": sys.version.split()[0],
            "providers": registry.availability(),
        }

    # ------------------------------------------------------- static assets
    # The app ships as plain ES modules with no build step and no hash in their
    # names, so /js/pages/album.js is the same URL forever.  Served without a
    # Cache-Control header - which is what StaticFiles does - a browser is free
    # to invent its own freshness from Last-Modified and keep serving a copy it
    # already has, without ever asking.  Chrome did exactly that: the album kept
    # posting to an endpoint that had since been renamed and answered "Method
    # Not Allowed", while the file on disk and the file the server returned were
    # both correct and every reload showed the same stale behaviour.  That is
    # unfalsifiable from the user's side and it cost a day.
    #
    # no-cache does NOT mean "do not store": the browser still caches and still
    # gets a 304 on an unchanged file thanks to the ETag, so the traffic is a
    # conditional request, not a re-download.  It only removes the browser's
    # licence to skip asking.  Applied to the shell (html/js/css) alone; icons
    # and generated images keep the default, because those really are immutable
    # and are the only ones big enough for it to matter.
    @app.middleware("http")
    async def revalidate_app_shell(request, call_next):
        response = await call_next(request)
        path = request.url.path
        if (path in ("/", "/sw.js", "/manifest.webmanifest")
                or path.startswith("/js/") or path.startswith("/css/")
                or not path.startswith(("/api/", "/icons/", "/assets/"))):
            response.headers.setdefault("Cache-Control", "no-cache")
        return response

    if FRONTEND_DIR.exists():
        app.mount("/css", StaticFiles(directory=FRONTEND_DIR / "css"), name="css")
        app.mount("/js", StaticFiles(directory=FRONTEND_DIR / "js"), name="js")
        for extra in ("icons", "assets"):
            d = FRONTEND_DIR / extra
            if d.exists():
                app.mount(f"/{extra}", StaticFiles(directory=d), name=extra)

        @app.get("/manifest.webmanifest")
        def manifest():
            p = FRONTEND_DIR / "manifest.webmanifest"
            if p.exists():
                return FileResponse(p, media_type="application/manifest+json")
            return JSONResponse({"detail": "not found"}, status_code=404)

        @app.get("/sw.js")
        def service_worker():
            p = FRONTEND_DIR / "sw.js"
            if p.exists():
                return FileResponse(p, media_type="application/javascript")
            return PlainTextResponse("", media_type="application/javascript")

        @app.get("/")
        @app.get("/{full_path:path}")
        def spa(full_path: str = ""):
            """Single page app: every non /api path returns index.html."""
            if full_path.startswith("api/"):
                return JSONResponse({"detail": "No encontrado"}, status_code=404)
            index = FRONTEND_DIR / "index.html"
            if index.exists():
                return FileResponse(index, media_type="text/html")
            return PlainTextResponse("frontend/index.html missing", status_code=500)

    return app


app = create_app()

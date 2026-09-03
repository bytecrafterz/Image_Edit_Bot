"""Launcher.

Prints, in Spanish, the address to open on this machine and the address to open
from a phone on the same network - because the whole point is that the client
uses this from her iPhone without being told what an IP address is.
"""
from __future__ import annotations

import argparse
import socket
import time
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import SETTINGS, key_status  # noqa: E402


def lan_ip() -> str:
    """The address of this machine on the local network, without asking anyone."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("10.255.255.255", 1))   # no packet is actually sent
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def banner(host: str, port: int) -> None:
    ip = lan_ip()
    keys = key_status()
    line = "=" * 66
    print(line)
    print(f"  {SETTINGS.app_name} v{SETTINGS.version}")
    print(line)
    print("  En este ordenador:   http://localhost:%d" % port)
    if host in ("0.0.0.0", "::"):
        print("  Desde el movil:      http://%s:%d" % (ip, port))
        print("  (el movil tiene que estar en la misma red wifi)")
    print()
    missing = [name for name in ("anthropic", "fal") if not keys[name]["present"]]
    if missing:
        print("  Sin claves de: %s" % ", ".join(missing))
        print("  El sistema funcionara con el motor local gratuito, que")
        print("  transforma tus fotos sin coste. Puedes anadir las claves")
        print("  mas tarde en Ajustes.")
    else:
        print("  Claves configuradas: Anthropic y fal.ai")
    print()
    print("  Para detener el servidor: Ctrl+C")
    print(line, flush=True)


def port_taken(host: str, port: int) -> bool:
    """Can we actually claim this port?

    Worth checking before uvicorn starts: on this machine nginx holds 8080, and
    when two servers claim one port on Windows the other one can keep answering
    localhost, so the app appears to start while the browser shows a different
    site entirely.

    This asks by BINDING, not by connecting.  Connecting looked simpler and was
    wrong: a socket left behind by a process that has already exited, or a
    browser keep-alive to the server we just stopped, still accepts a connection
    for a while.  The guard then refused to start on a perfectly free port -
    turning an ordinary restart into "el puerto ya esta ocupado" - which is
    precisely the failure it existed to prevent.  A bind attempt answers the
    only question that matters: can uvicorn have this port or not.
    """
    bind_host = "" if str(host) in ("0.0.0.0", "::", "") else str(host)
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((bind_host, int(port)))
        return False
    except OSError:
        return True
    finally:
        probe.close()


def port_shadowed(port: int) -> bool:
    """Is someone already answering on localhost, even if we CAN bind?

    Windows lets us bind 0.0.0.0:8080 while another server holds
    127.0.0.1:8080; we start happily and the browser keeps reaching the other
    one.  Binding succeeds, so the bind test alone says nothing is wrong - this
    is the case that made the app look broken for no visible reason.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.6)
    try:
        return probe.connect_ex(("127.0.0.1", int(port))) == 0
    except OSError:
        return False
    finally:
        probe.close()


def free_port_near(port: int) -> int:
    """First free port at or after ``port``, so the advice is actionable."""
    for candidate in range(int(port), int(port) + 60):
        if not port_taken("127.0.0.1", candidate):
            return candidate
    return int(port)


def main() -> int:
    parser = argparse.ArgumentParser(description="Arranca Photo Robot")
    parser.add_argument("--host", default=SETTINGS.host)
    parser.add_argument("--port", type=int, default=SETTINGS.port)
    parser.add_argument("--reload", dest="reload", action="store_true",
                        default=True,
                        help="recarga automatica al cambiar el codigo (por defecto)")
    parser.add_argument("--no-reload", dest="reload", action="store_false",
                        help="no recargar; usalo cuando este en produccion")
    args = parser.parse_args()

    # Wait a moment before believing the port is taken.  Restarting the server
    # is the commonest thing anyone does here, and for a second or two after the
    # old process dies its socket is still in TIME_WAIT - refusing on that first
    # look turns a normal restart into a confusing "port occupied" error, which
    # is exactly what happened the first time this guard shipped.
    if port_taken(args.host, args.port):
        for _ in range(10):
            time.sleep(0.5)
            if not port_taken(args.host, args.port):
                break

    if port_taken(args.host, args.port):
        alternative = free_port_near(args.port + 1)
        print("=" * 66)
        print("  El puerto %d ya esta ocupado por otro programa." % args.port)
        print("  Si arrancamos igualmente, el navegador puede mostrarte esa otra")
        print("  web en lugar de Photo Robot.")
        print()
        print("  Arranca en un puerto libre:")
        print("      scripts\\start.ps1 --port %d" % alternative)
        print("=" * 66, flush=True)
        return 2

    if port_shadowed(args.port):
        # We can bind, but something else already answers on localhost, so the
        # browser would keep reaching it instead of us.  Worth a warning, not a
        # refusal: the LAN address will still be ours.
        print("=" * 66)
        print("  AVISO: otro programa ya responde en localhost:%d." % args.port)
        print("  Podemos arrancar, pero el navegador puede seguir mostrando esa")
        print("  otra web. Si ves algo que no es Photo Robot, usa otro puerto:")
        print("      scripts\\start.ps1 --port %d" % free_port_near(args.port + 1))
        print("=" * 66, flush=True)

    banner(args.host, args.port)
    if args.reload:
        print("  Recarga automatica activada: al guardar un archivo .py el")
        print("  servidor se reinicia solo. Un trabajo en curso se cancela,")
        print("  asi que no guardes codigo mientras se generan imagenes.")
        print("  Para desactivarla: --no-reload")
        print("=" * 66, flush=True)

    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=args.host, port=args.port,
        reload=args.reload,
        # Watch the source and NOTHING else.  Uvicorn's default is the working
        # directory, which here contains data/ - every generated image, every
        # thumbnail and the SQLite write-ahead log land in there, so the default
        # would restart the server in a loop in the middle of a run and kill the
        # very job that was writing the files.
        reload_dirs=[str(BACKEND_DIR / "app")] if args.reload else None,
        reload_includes=["*.py"] if args.reload else None,
        reload_excludes=["*__pycache__*", "*.pyc", "*.sqlite3*", "*.log",
                         "*.jpg", "*.jpeg", "*.png"] if args.reload else None,
        log_level="info", access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

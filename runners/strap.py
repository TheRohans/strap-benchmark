"""Strap harness runner via IRC."""
import logging
import os
import socket
import threading
import time
from queue import Empty, Queue

log = logging.getLogger("strap")


class StrapRunner:
    def __init__(
        self,
        host:          str        = os.environ.get("STRAP_HOST",     "localhost"),
        port:          int        = int(os.environ.get("STRAP_PORT", "6667")),
        password:      str | None = os.environ.get("STRAP_PASSWORD") or None,
        channel:       str        = os.environ.get("STRAP_CHANNEL",  "#main"),
        silence_secs:  float      = 4.0,
        timeout_secs:  float      = 60.0,
    ):
        self.host = host
        self.port = port
        self.password = password
        self.channel = channel
        self.silence_secs = silence_secs
        self.timeout_secs = timeout_secs

        self._sock: socket.socket | None = None
        self._responses: Queue[str] = Queue()
        self._connected = False

    def connect(self) -> None:
        log.info("connecting to %s:%d", self.host, self.port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self.host, self.port))
        self._connected = True

        t = threading.Thread(target=self._read_loop, daemon=True)
        t.start()

        if self.password:
            log.debug("→ PASS ***")
            self._sock.sendall((f"PASS {self.password}\r\n").encode())
        self._send("NICK benchmarkbot")
        self._send("USER benchmarkbot 0 * :Benchmark Runner")
        time.sleep(0.5)  # wait for server welcome + auto-join #main
        log.info("connected, session on %s", self.channel)

    def reset(self) -> None:
        log.debug("resetting session (.clear)")
        self._drain()
        self._send(f"PRIVMSG {self.channel} :.clear")
        time.sleep(1.0)
        self._drain()

    def close(self) -> None:
        log.info("closing connection")
        self._connected = False
        if self._sock:
            try:
                self._send("QUIT :benchmark done")
            except Exception:
                pass
            self._sock.close()

    def run(self, task: dict) -> str:
        turns = self._build_turns(task)
        log.debug("task %s — %d turn(s)", task["id"], len(turns))

        for i, turn in enumerate(turns[:-1]):
            log.debug("  sending turn %d/%d: %s", i + 1, len(turns), turn[:80])
            self._drain()
            self._send(f"PRIVMSG {self.channel} :{turn}")
            self._collect(label=f"turn-{i+1}")

        final = turns[-1]
        log.debug("  sending final turn: %s", final[:80])
        self._drain()
        self._send(f"PRIVMSG {self.channel} :{final}")
        return self._collect(label="final")

    # ------------------------------------------------------------------

    def _send(self, msg: str) -> None:
        log.debug("→ %s", msg)
        assert self._sock is not None
        self._sock.sendall((msg + "\r\n").encode())

    def _read_loop(self) -> None:
        buf = ""
        while self._connected:
            try:
                data = self._sock.recv(4096).decode(errors="replace")  # type: ignore[union-attr]
            except OSError:
                log.warning("socket closed in read loop")
                break
            buf += data
            while "\r\n" in buf:
                line, buf = buf.split("\r\n", 1)
                self._handle_line(line)

    def _handle_line(self, line: str) -> None:
        if line.startswith("PING"):
            log.debug("← PING, sending PONG")
            self._send("PONG" + line[4:])
            return
        if f"PRIVMSG {self.channel} :" in line:
            nick = line.split("!", 1)[0].lstrip(":")
            if nick == "benchmarkbot":
                return
            msg = line.split(f"PRIVMSG {self.channel} :", 1)[-1]
            log.debug("← [%s] %s", nick, msg[:120])
            self._responses.put(msg)
        else:
            log.debug("← %s", line)

    def _drain(self) -> None:
        drained = 0
        while not self._responses.empty():
            try:
                self._responses.get_nowait()
                drained += 1
            except Empty:
                break
        if drained:
            log.debug("drained %d stale message(s)", drained)

    def _collect(self, label: str = "") -> str:
        """Accumulate response chunks until silence_secs of quiet."""
        parts: list[str] = []
        last_received = time.time()
        start = last_received

        while True:
            try:
                msg = self._responses.get(timeout=0.5)
                parts.append(msg)
                last_received = time.time()
            except Empty:
                silent_for = time.time() - last_received
                elapsed    = time.time() - start

                if parts and silent_for >= self.silence_secs:
                    log.debug(
                        "collect[%s] done — %d chunk(s), %.1fs silence, %.1fs total",
                        label, len(parts), silent_for, elapsed,
                    )
                    break
                if silent_for >= self.timeout_secs:
                    log.warning(
                        "collect[%s] TIMEOUT after %.1fs — %d chunk(s) received",
                        label, elapsed, len(parts),
                    )
                    break

        return "\n".join(parts)

    def _build_turns(self, task: dict) -> list[str]:
        if task["type"] == "reasoning":
            return [task["prompt"]]
        return [t["content"] for t in task["turns"] if t["role"] == "user"]
